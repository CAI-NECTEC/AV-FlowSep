import torch
import random
import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.model import AVFlowModel
from src.data_module import MelSpecDataModule

import warnings
warnings.filterwarnings(action='ignore')

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed) 
    random.seed(seed) 

seed_everything(112)

# ==================================
# ============ Settings ============
# ==================================

save_path = "save_path"

backbone = "avss_dit"
ode = "flowmatching" # flowmatching | without_latent | with_latent
lr = 1e-4
num_eval_files = 50
loss_type = "mse"
pretrained_talknet = "pretrain_talknet_model_path"
vocoder_path = "vocoder_path"

epochs = 1000
accelerator = "gpu"
strategy = "ddp_find_unused_parameters_true" # ddp, auto, ddp_notebook, ddp_find_unused_parameters_true
num_nodes = 16
devices = 4 # "auto"

data_module_kwargs = {
    "train_ann_file": "train_dataset_path",
    "val_ann_file": "eval_dataset_path",
    "test_ann_file": "test_dataset_path",
    "batch_size": 8,
    "num_workers": 4
}

# ==================================
# ==================================
# ==================================

model = AVFlowModel(
    backbone = backbone,
    # depth = 12,
    # hidden_dim = 512,
    # num_heads = 8,
    ode = ode, # flowmatching | without_latent | with_latent
    lr = lr,
    num_eval_files = num_eval_files,
    loss_type = loss_type,
    pretrained_talknet = pretrained_talknet,
    vocoder_path = vocoder_path,
    data_module_cls = MelSpecDataModule,
    **data_module_kwargs
)

logger = CSVLogger(save_dir=save_path, name="logs")

callbacks = ModelCheckpoint(
    dirpath=save_path,
    filename="model-{epoch:02d}",
    save_top_k=1, # -1 save all every epoch
    # every_n_epochs=10,
    save_last=False,
    save_on_train_epoch_end=True
)

ckpt_pesq = ModelCheckpoint(
    dirpath=save_path,
    filename="best-pesq-{epoch:02d}-{pesq:.4f}",
    monitor="pesq",
    mode="max",
    save_top_k=1,
    save_last=False
)

ckpt_si_sdr = ModelCheckpoint(
    dirpath=save_path,
    filename="best-si_sdr-{epoch:02d}-{si_sdr:.4f}",
    monitor="si_sdr",
    mode="max",
    save_top_k=1,
    save_last=False
)

ckpt_loss = ModelCheckpoint(
    dirpath=save_path,
    filename="best-loss-{epoch:02d}-{valid_loss:.4f}",
    monitor="valid_loss",
    mode="min",
    save_top_k=1,
    save_last=False
)

trainer = pl.Trainer(
    accelerator=accelerator,
    strategy=strategy, # ddp, auto, ddp_notebook
    devices=devices,
    num_nodes=num_nodes,
    max_epochs=epochs,
    accumulate_grad_batches=1,
    logger=logger,
    callbacks=[callbacks, ckpt_pesq, ckpt_si_sdr, ckpt_loss]
)

trainer.fit(model)