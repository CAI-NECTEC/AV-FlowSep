import os
import soundfile as sf
import time

import torch
import random
import numpy as np

import pytorch_lightning as pl
from src.model import AVFlowModel
from src.data_module import MelSpecDataModule

import torch.nn.functional as F

from vocos import Vocos
from vocos.feature_extractors import (
    MelSpectrogramFeatures, FeatureExtractor, EncodecFeatures
)
from scipy.signal import resample_poly

device = "cuda" if torch.cuda.is_available() else "cpu"

# =================================================================================================

ckpt = "av-flowsep_model_path"

model = AVFlowModel.load_from_checkpoint(ckpt)
model.eval() # no_ema=False
model = model.to(device)
# =================================================================================================

vocos_path = "vocoder_path"
vocos_cls = Vocos
vocos = vocos_cls.from_hparams(os.path.join(vocos_path, "config.yaml"))
state_dict = torch.load(os.path.join(vocos_path, "pytorch_model.bin"), map_location="cpu")
if isinstance(vocos.feature_extractor, EncodecFeatures):
    encodec_parameters = {
        "feature_extractor.encodec." + key: value
        for key, value in vocos.feature_extractor.encodec.state_dict().items()
    }
    state_dict.update(encodec_parameters)
vocos.load_state_dict(state_dict)
vocos.eval()

# =================================================================================================

data_module = MelSpecDataModule(
    train_ann_file="train_datasets_path",
    val_ann_file="eval_datasets_path",
    test_ann_file="test_datasets_path",
    batch_size=8,
    num_workers=4
)

data_module.setup()

start_time = time.time()

# Settings
sr = 24000
N = 5

# =================================================================================================

i = 0

x, y, visual_features = data_module.test_set.__getitem__(i, return_wave=True)
visual_features = torch.Tensor(visual_features).unsqueeze(0).cuda()

Y = model.data_module.feature_extractor(y).unsqueeze(0)

sampler = model.get_ode_sampler(
        odesolver_name="euler",
        y=Y.cuda(), 
        context=visual_features,
        N=N
)
mel_sample, _ = sampler()   

x_hat = vocos.decode(mel_sample.squeeze(0).cpu()) 

x_hat = x_hat.squeeze().cpu().numpy()
x = x.squeeze().cpu().numpy()
y = y.squeeze().cpu().numpy()

out_dir = "./examples"
os.makedirs(out_dir, exist_ok=True)

sf.write(f"{out_dir}/clean_x.wav", x, sr)
sf.write(f"{out_dir}/mixture_y.wav", y, sr)
sf.write(f"{out_dir}/enhanced_x_hat.wav", x_hat, sr)

end_time = time.time()
print(f"Execution time: {end_time - start_time:.6f} seconds")