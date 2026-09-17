mport argparse
import warnings

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from src.data_module import MelSpecDataModule
from src.model import AVFlowModel

warnings.filterwarnings("ignore")


def set_seed(seed: int) -> None:
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_callbacks(save_path: str) -> list:
    """Checkpoint the latest epoch plus the best model for each metric."""

    def best(metric: str, mode: str) -> ModelCheckpoint:
        return ModelCheckpoint(
            dirpath=save_path,
            filename=f"best-{metric}-{{epoch:02d}}-{{{metric}:.4f}}",
            monitor=metric,
            mode=mode,
            save_top_k=1,
            save_last=False,
        )

    latest = ModelCheckpoint(
        dirpath=save_path,
        filename="model-{epoch:02d}",
        save_top_k=1,  # -1 to keep every epoch
        save_last=False,
        save_on_train_epoch_end=True,
    )

    return [
        latest,
        best("pesq", "max"),
        best("valid_loss", "min"),
    ]

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)

    # Paths
    p.add_argument("--save-path", required=True, help="Directory for checkpoints and logs")
    p.add_argument("--train-ann", required=True, help="Training annotation CSV")
    p.add_argument("--val-ann", required=True, help="Validation annotation CSV")
    p.add_argument("--test-ann", default="", help="Test annotation CSV")
    p.add_argument("--pretrained-talknet", required=True, help="Pretrained TalkNet checkpoint")
    p.add_argument("--vocoder-path", required=True, help="Vocos vocoder directory")

    # Model
    p.add_argument("--backbone", default="avss_dit")
    p.add_argument("--ode", default="flowmatching",
                   choices=["flowmatching", "without_latent", "with_latent"])
    p.add_argument("--loss-type", default="mse")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-eval-files", type=int, default=50)

    # Data
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)

    # Trainer
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--accelerator", default="gpu")
    p.add_argument("--strategy", default="ddp_find_unused_parameters_true",
                   help="ddp | auto | ddp_notebook | ddp_find_unused_parameters_true")
    p.add_argument("--num-nodes", type=int, default=4)
    p.add_argument("--devices", default="4", help='GPUs per node, or "auto"')
    p.add_argument("--accumulate-grad-batches", type=int, default=1)
    p.add_argument("--seed", type=int, default=112)

    args = p.parse_args()
    if args.devices != "auto":
        args.devices = int(args.devices)
    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    model = AVFlowModel(
        backbone=args.backbone,
        ode=args.ode,
        lr=args.lr,
        num_eval_files=args.num_eval_files,
        loss_type=args.loss_type,
        pretrained_talknet=args.pretrained_talknet,
        vocoder_path=args.vocoder_path,
        data_module_cls=MelSpecDataModule,
        train_ann_file=args.train_ann,
        val_ann_file=args.val_ann,
        test_ann_file=args.test_ann,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        strategy=args.strategy,
        devices=args.devices,
        num_nodes=args.num_nodes,
        max_epochs=args.epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        logger=CSVLogger(save_dir=args.save_path, name="logs"),
        callbacks=build_callbacks(args.save_path),
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
