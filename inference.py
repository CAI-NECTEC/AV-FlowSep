import argparse
import os

import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from vocos import Vocos
from vocos.feature_extractors import EncodecFeatures

from src.data_module import MelSpecDataModule
from src.model import AVFlowModel

REPO_ID = "nectec/av-flowsep"
CKPT_NAME = "av-flowsep-best-pesq.ckpt"
VOCOS_DIR = "vocos-mel-24khz"
SAMPLE_RATE = 24000


def get_repo_dir(repo_id: str) -> str:
    """Return the local snapshot path, downloading only if not cached."""
    try:
        repo_dir = snapshot_download(repo_id=repo_id, local_files_only=True)
        print(f"Using cached model: {repo_dir}")
    except LocalEntryNotFoundError:
        print("Cache not found, downloading...")
        repo_dir = snapshot_download(repo_id=repo_id)
        print(f"Downloaded to: {repo_dir}")
    return repo_dir


def load_model(repo_dir: str, device: torch.device) -> AVFlowModel:
    ckpt = os.path.join(repo_dir, CKPT_NAME)
    model = AVFlowModel.load_from_checkpoint(ckpt, pretrained_talknet=False)
    return model.eval().to(device)


def load_vocoder(repo_dir: str, device: torch.device) -> Vocos:
    vocos_path = os.path.join(repo_dir, VOCOS_DIR)
    vocos = Vocos.from_hparams(os.path.join(vocos_path, "config.yaml"))
    state_dict = torch.load(
        os.path.join(vocos_path, "pytorch_model.bin"), map_location="cpu"
    )

    if isinstance(vocos.feature_extractor, EncodecFeatures):
        encodec_params = {
            f"feature_extractor.encodec.{k}": v
            for k, v in vocos.feature_extractor.encodec.state_dict().items()
        }
        state_dict.update(encodec_params)

    vocos.load_state_dict(state_dict)
    return vocos.eval().to(device)


def load_test_set(csv_path: str):
    data_module = MelSpecDataModule(
        train_ann_file="",
        val_ann_file="",
        test_ann_file=csv_path,
        batch_size=1,
        num_workers=4,
    )
    data_module.setup(stage="test")
    return data_module.test_set


@torch.inference_mode()
def separate(model, vocos, test_set, index: int, n_steps: int, device):
    """Run the ODE sampler on one sample; return (clean, mixture, estimate) as numpy."""
    x, y, visual_features = test_set.__getitem__(index, return_wave=True)

    visual = torch.as_tensor(visual_features, dtype=torch.float32).unsqueeze(0).to(device)
    y_mel = model.data_module.feature_extractor(y).unsqueeze(0).to(device)

    sampler = model.get_ode_sampler(
        odesolver_name="euler",
        y=y_mel,
        context=visual,
        N=n_steps,
    )
    mel_sample, _ = sampler()
    x_hat = vocos.decode(mel_sample.squeeze(0))

    to_np = lambda t: t.squeeze().detach().cpu().numpy()
    return to_np(x), to_np(y), to_np(x_hat)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default="./data/test.csv",
        help="Test annotation CSV",
    )
    parser.add_argument("--index", type=int, default=0, help="Sample index in the test set")
    parser.add_argument("--steps", type=int, default=5, help="Number of ODE solver steps")
    parser.add_argument("--out-dir", default="./examples", help="Where to write wav files")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    repo_dir = get_repo_dir(REPO_ID)
    model = load_model(repo_dir, device)
    vocos = load_vocoder(repo_dir, device)
    test_set = load_test_set(args.csv)

    x, y, x_hat = separate(model, vocos, test_set, args.index, args.steps, device)

    os.makedirs(args.out_dir, exist_ok=True)
    sf.write(os.path.join(args.out_dir, "clean_x.wav"), x, SAMPLE_RATE)
    sf.write(os.path.join(args.out_dir, "mixture_y.wav"), y, SAMPLE_RATE)
    sf.write(os.path.join(args.out_dir, "enhanced_x_hat.wav"), x_hat, SAMPLE_RATE)

    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
