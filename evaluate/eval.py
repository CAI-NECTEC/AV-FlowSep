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
from tqdm import tqdm

sr = 24000
N = 1
save_dir = "save_path"
os.makedirs(save_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ================================================
# ============== Load AV Flow Model ==============
# ================================================

ckpt = "av-flowsep_model_path"
model = AVFlowModel.load_from_checkpoint(ckpt)
model.eval() # no_ema=False
model = model.to(device)

# ================================================
# ================= Load Vocoder =================
# ================================================

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

# ===============================================
# ================= Data Module =================
# ===============================================

data_module = MelSpecDataModule(
    train_ann_file="train_datasets_path",
    val_ann_file="eval_datasets_path",
    test_ann_file="test_datasets_path",
    batch_size=8,
    num_workers=4
)

data_module.setup()

# ===============================================
# =================== Evalute ===================
# ===============================================

df = data_module.test_set.df
for i in tqdm(range(len(df))):
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

    filename = df.clean_wav[i].split("/")[-1][:-4] + f"_{i}.wav"
    save_path = os.path.join(save_dir, filename)
    df.at[i, "pred_wav"] = save_path
    sf.write(save_path, x_hat, sr)

df.to_csv(os.path.join(save_dir, "result.csv"), index=False)