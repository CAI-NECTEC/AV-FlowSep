import os
import time
import torch
import torchaudio
import librosa
import numpy as np
import pandas as pd
from pesq import pesq
# from pypesq import pesq
from pystoi import stoi
# from UTMOS import UTMOSScore
from DNSMOS import deep_noise_suppression_mean_opinion_score
from periodicity import calculate_periodicity_metrics
from pymcd.mcd import Calculate_MCD
# import IPython.display as ipd
from tqdm.auto import tqdm
tqdm.pandas()

# ===================================
# ============== Utils ==============
# ===================================

def si_sdr(s, s_hat):
    alpha = np.dot(s_hat, s)/np.linalg.norm(s)**2   
    sdr = 10*np.log10(np.linalg.norm(alpha*s)**2/np.linalg.norm(
        alpha*s - s_hat)**2)
    return sdr

def extract_mcd(audio_ref, audio_deg, mode="plain", sr=16000):
    mcd_toolbox = Calculate_MCD(MCD_mode=mode) # plain, dtw_sl, dtw
    if sr != None:
        mcd_toolbox.SAMPLING_RATE = sr
    mcd_value = mcd_toolbox.calculate_mcd(audio_ref, audio_deg)
    return mcd_value
    
def load_align_audio(audio_ref, audio_deg, sr=16000, method="cut"):

    # Audio length alignment
    if len(audio_ref) != len(audio_deg):
        if method == "cut":
            length = min(len(audio_ref), len(audio_deg))
            audio_ref = audio_ref[:length]
            audio_deg = audio_deg[:length]
        elif method == "dtw":
            _, wp = librosa.sequence.dtw(
                audio_ref, audio_deg, metric="euclidean", backtrack=True
                )
            audio_ref_new = []
            audio_deg_new = []
            for i in range(wp.shape[0]):
                ref_index = wp[i][0]
                deg_index = wp[i][1]
                audio_ref_new.append(audio_ref[ref_index])
                audio_deg_new.append(audio_deg[deg_index])
            audio_ref = np.array(audio_ref_new)
            audio_deg = np.array(audio_deg_new)
            assert len(audio_ref) == len(audio_deg)     

    return audio_ref, audio_deg

# ==================================
# ============= Define =============
# ==================================

# Define
sr = 16000
align_mode = "cut" # cut, dtw

csv_path = "/project/lt200299-aicook/earth/interspeech2026/results/mossformer2-lrs2-audioset/result.csv"

save_path = "/project/lt200299-aicook/earth/interspeech2026/av-flowsep/metrics/results/mossformer2-lrs2-audioset.csv"

os.makedirs("./results", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# utmos_model = UTMOSScore(device=device)

df = pd.read_csv(csv_path)

# ===================================
# ============= Process =============
# ===================================

def process_row(row):

    clean_wav = row["clean_wav"]
    pred_wav = row["pred_wav"]

    audio_clean, _ = librosa.load(clean_wav, sr=sr)
    audio_pred, _ = librosa.load(pred_wav, sr=sr)

    results = {}

    # # UTMOS
    # _utmos = utmos_model.score(
    #     torch.Tensor(audio_pred).unsqueeze(0).to(device)
    # ).mean()
    # results["utmos"] = float(_utmos.item())

    # DNSMOS
    _, _dnsmos_sig, _dnsmos_bak, _dnsmos_ovrl = \
        deep_noise_suppression_mean_opinion_score(
            torch.Tensor(audio_pred).unsqueeze(0), sr, False, num_threads=8
        )[0]

    results["dnsmos_sig"] = float(_dnsmos_sig.item())
    results["dnsmos_bak"] = float(_dnsmos_bak.item())
    results["dnsmos_ovrl"] = float(_dnsmos_ovrl.item())

    # MCD
    _mcd = extract_mcd(clean_wav, pred_wav, mode="dtw_sl", sr=sr)
    results["mcd"] = float(_mcd.item())

    # Align audio
    audio_clean, audio_pred = load_align_audio(
        audio_clean, audio_pred, sr=sr, method=align_mode
    )

    # # Periodicity
    # _periodicity_loss, _pitch_loss, _f1_score = calculate_periodicity_metrics(
    #     torch.Tensor(audio_clean).unsqueeze(0),
    #     torch.Tensor(audio_pred).unsqueeze(0)
    # )
    # results["periodicity_loss"] = float(_periodicity_loss.item())
    # results["pitch_loss"] = float(_pitch_loss.item())
    # results["f1_score"] = float(_f1_score.item())

    # PESQ
    results["pesq"] = float(pesq(sr, audio_clean, audio_pred, "wb")) # nb, wb
    # results["pesq"] = pesq(audio_clean, audio_pred, sr)

    # SI-SDR
    results["si_sdr"] = float(si_sdr(audio_clean, audio_pred))

    # STOI
    results["stoi"] = float(stoi(audio_clean, audio_pred, sr, extended=True))

    return pd.Series(results)

df_results = df.progress_apply(process_row, axis=1)
df = pd.concat([df, df_results], axis=1)
df.to_csv(save_path, index=False)

# print(f"UTMOS: {df['utmos'].mean().item():.3f} ↑")
print(f"DNSMOS SIG: {df['dnsmos_sig'].mean().item():.3f} ↑")
print(f"DNSMOS BAK: {df['dnsmos_bak'].mean().item():.3f} ↑")
print(f"DNSMOS OVRL: {df['dnsmos_ovrl'].mean().item():.3f} ↑")
print(f"MCD: {df['mcd'].mean().item():.3f} ↓")
# print(f"Periodicity: {df['periodicity_loss'].mean().item():.3f} ↓")
# print(f"Pitch Loss: {df['pitch_loss'].mean().item():.3f} ↓")
# print(f"V/UV F1: {df['f1_score'].mean().item():.3f} ↑")
print(f"PESQ: {df['pesq'].mean().item():.3f} ↑")
print(f"SI SDR: {df['si_sdr'].mean().item():.3f} ↑")
print(f"STOI: {df['stoi'].mean().item():.3f} ↑")
