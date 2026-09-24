# AV-FlowSep: Audio-Visual Target Speaker Separation via Flow Matching

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://cai-nectec.github.io/AV-FlowSep/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/nectec/av-flowsep)
[![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-lightgrey)](https://www.isca-archive.org/interspeech_2026/tipaksorn26_interspeech.html#)
[![Interspeech 2026](https://img.shields.io/badge/Interspeech-2026-red)](https://www.isca-archive.org/interspeech_2026/tipaksorn26_interspeech.html#)

## Overview

AV-FlowSep separates a target speaker's voice from a mixture using visual cues from their face. It uses conditional flow matching to learn a straight transport path from the mixture mel-spectrogram to the clean one. The backbone is a Diffusion Transformer (DiT), and visual features come in through cross-attention. It produces high-quality speech **in a single inference step**, where diffusion-based methods need about 30.

<p align="center">
  <img src="https://raw.githubusercontent.com/CAI-NECTEC/cai-nectec.github.io/main/docs/AV-FlowSep/asset/image/Architecture.png" width="400" alt="Architecture">
</p>

The model has three parts:

- **Visual encoder:** the TalkNet-ASD frontend (3D conv, ResNet-18, V-TCN) on 25 FPS grayscale 112×112 face crops.
- **DiT vector field estimator:** the DiT-S configuration with 12 layers, 6 heads, and hidden size 384. The timestep enters through AdaLN and visual features through cross-attention.
- **Vocoder:** [Vocos](https://github.com/gemelo-ai/vocos) turns 100-bin log-mel features at 24 kHz back into a waveform.

## Results

These results use one inference step (53.6M parameters). LRS2 is evaluated zero-shot, without fine-tuning.

| Test set | DNSMOS OVRL ↑ | MCD ↓ | PESQ ↑ |
|---|---|---|---|
| VoxCeleb2-2Mix | 2.484 | 5.985 | 2.751 |
| VoxCeleb2-AudioSet | 2.695 | 4.400 | 2.952 |
| LRS2-2Mix (zero-shot) | 2.364 | 7.276 | 2.404 |
| LRS2-AudioSet (zero-shot) | 2.827 | 4.190 | 2.824 |

The paper has the full comparison with SepFormer, VisualVoice, AV-MossFormer2, and AVDiffuSS.

## Installation

```bash
git clone https://github.com/CAI-NECTEC/AV-FlowSep.git
cd AV-FlowSep
pip install -r requirements.txt
```

> **Note:** keep NumPy on 1.x for compatibility with older dependencies:
> `pip install "numpy<2" "opencv-python<4.12"`

## Pretrained Models

Weights are hosted on [Hugging Face](https://huggingface.co/nectec/av-flowsep):

- `av-flowsep-best-pesq.ckpt`: the AV-FlowSep checkpoint
- `vocos-mel-24khz/`: the Vocos vocoder

The inference script downloads both automatically and reuses the local cache on later runs.

## Data Format

The data module reads a CSV with one row per target speaker:

```csv
mp4,clean_wav,mix_wav
visual_target_path.mp4,clean_audio_path.wav,mix_audio_path.wav
```

- `mp4`: video of the target speaker
- `clean_wav`: the target speaker's clean reference audio
- `mix_wav`: the mixture

A sample test file is in `./data/test.csv`.

## Inference

```bash
python inference.py --index 0 --steps 5 --out-dir ./examples
```

This writes `clean_x.wav`, `mixture_y.wav`, and `enhanced_x_hat.wav` to `--out-dir`. Use `--steps 1` for the fastest single-step inference.

## Training

```bash
python train.py \
  --save-path ./ckpts \
  --train-ann data/train.csv \
  --val-ann data/val.csv \
  --pretrained-talknet pretrained/talknet.model \
  --vocoder-path pretrained/vocos-mel-24khz
```

By default training runs 200 epochs with batch size 8, Adam at lr 1e-4, and DDP on 4 nodes × 4 GPUs. Override the setup with `--num-nodes` and `--devices`. Checkpoints are saved for the best PESQ, and validation loss. The pretrained visual frontend comes from [TalkNet-ASD](https://github.com/TaoRuijie/TalkNet-ASD).

## Citation

The paper link will be added soon. In the meantime, please cite:

```bibtex
@inproceedings{tipaksorn2026avflowsep,
  title     = {AV-FlowSep: Audio-Visual Target Speaker Separation via Flow Matching},
  author    = {Pattara Tipaksorn and Wayupuk Sommuang and Kwanchiva Thangthai},
  booktitle = {Proc. Interspeech},
  year      = {2026}
}
```

## Acknowledgements

This work was supported by NECTEC, NSTDA, Thailand, under the CAI research team. Computing resources were provided by ThaiSC on the LANTA supercomputer. We build on [TalkNet-ASD](https://github.com/TaoRuijie/TalkNet-ASD), [DiT](https://github.com/facebookresearch/DiT), and [Vocos](https://github.com/gemelo-ai/vocos).
