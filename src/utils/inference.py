import os
import torch
import torch.nn.functional as F
from pesq import pesq
from pystoi import stoi
from src.utils.other import si_sdr, pad_spec
from vocos import Vocos
from vocos.feature_extractors import (
    MelSpectrogramFeatures, FeatureExtractor, EncodecFeatures
)
from scipy.signal import resample_poly

def load_vocoder(vocos_path: str, device: str = "cpu"):
    vocos = Vocos.from_hparams(os.path.join(vocos_path, "config.yaml"))
    state_dict = torch.load(os.path.join(vocos_path, "pytorch_model.bin"), map_location="cpu")
    if isinstance(vocos.feature_extractor, EncodecFeatures):
        encodec_parameters = {
            "feature_extractor.encodec." + key: value
            for key, value in vocos.feature_extractor.encodec.state_dict().items()
        }
        state_dict.update(encodec_parameters)
    vocos.load_state_dict(state_dict)
    vocos.eval()

    return vocos

# Settings
sr = 16000
N = 5

def evaluate_model(model, num_eval_files, vocoder_path):
    _pesq = 0
    _si_sdr = 0
    _estoi = 0

    vocos = load_vocoder(vocoder_path)

    for i in range(num_eval_files):
        x, y, visual_features = model.data_module.valid_set.__getitem__(i, return_wave=True)
        visual_features = torch.Tensor(visual_features).unsqueeze(0).cuda()

        X = model.data_module.feature_extractor(x).unsqueeze(0)
        Y = model.data_module.feature_extractor(y).unsqueeze(0)
        
        sampler = model.get_ode_sampler(
                odesolver_name="euler",
                y=Y.cuda(), 
                context=visual_features,
                N=N
        )
        mel_sample, _ = sampler()   
        
        x_hat = vocos.decode(mel_sample.squeeze(0).cpu())
        x = vocos.decode(X.squeeze(0).cpu())
        
        x_hat = x_hat.squeeze().cpu().numpy()
        x = x.squeeze().cpu().numpy()

        min_len = min(len(x_hat), len(x))
        x_hat = x_hat[:min_len]
        x = x[:min_len]

        # 24k → 16k  (ratio = 2/3)
        x_hat = resample_poly(x_hat, up=2, down=3)
        x = resample_poly(x, up=2, down=3)

        _si_sdr += si_sdr(x, x_hat)
        _pesq += pesq(sr, x, x_hat, 'wb') 
        _estoi += stoi(x, x_hat, sr, extended=True)
        
    return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files
