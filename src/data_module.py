import os
import cv2
import math
import pandas as pd
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from vocos import Vocos
from vocos.feature_extractors import MelSpectrogramFeatures

class MelSpec(Dataset):
    def __init__(
        self, 
        ann_file, 
        shuffle_spec=True,
        sampling_rate=24000,
        frame_rate=25,
        feature_extractor=None,
        **ignored_kwargs
        
    ):
        super().__init__()
        self.df = pd.read_csv(ann_file)

        self.video_files = self.df["mp4"].tolist()
        self.audio_clean_files = self.df["clean_wav"].tolist()
        self.audio_mix_files = self.df["mix_wav"].tolist()
        self.shuffle_spec = shuffle_spec
        self.sampling_rate = sampling_rate
        self.frame_rate = frame_rate
        self.chunk_size = sampling_rate * 2.04
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.audio_clean_files)
        
    def load_audio(self, path, target_sr=24000, mono=True):
        y, sr = torchaudio.load(path)  
        if mono and y.size(0) > 1:
            y = y.mean(dim=0, keepdim=True)
        if sr != target_sr:
            y = torchaudio.functional.resample(
                y,
                orig_freq=sr,
                new_freq=target_sr
            )
            sr = target_sr
    
        return y, sr

    def load_audio_pair(self, clean_path, mix_path):
        x, sr1 = self.load_audio(clean_path, target_sr=self.sampling_rate, mono=True)
        y, sr2 = self.load_audio(mix_path, target_sr=self.sampling_rate, mono=True)
    
        min_len = min(x.size(-1), y.size(-1))
        x = x[..., :min_len]
        y = y[..., :min_len]
    
        audio_size = min_len
        pad = max(int(self.chunk_size - audio_size), 0)
    
        if pad == 0:
            if self.shuffle_spec:
                start = torch.randint(
                    low=0,
                    high=audio_size - int(self.chunk_size) + 1,
                    size=(1,)
                ).item()
            else:
                start = (audio_size - int(self.chunk_size)) // 2
    
            x = x[..., start : start + int(self.chunk_size)]
            y = y[..., start : start + int(self.chunk_size)]
    
        else:
            x = F.pad(x, (0, pad), mode="constant", value=0.0)
            y = F.pad(y, (0, pad), mode="constant", value=0.0)
            start = 0
    
        return x, y, start

    def load_video(self, file_path, start_frame):
        video_start = int(start_frame // self.sampling_rate * self.frame_rate)
        N = math.ceil(self.chunk_size / self.sampling_rate * self.frame_rate)
        
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return None
                
        fps = cap.get(cv2.CAP_PROP_FPS)
        if int(fps) != self.frame_rate:
            return None
    
        frames = []
        for i in range(video_start + N): # 51 frames around 2.04 sec
            ret, img = cap.read()
            if i < video_start:
                continue
            if ret:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (112, 112))
                frames.append(img)
            else:
                frames = np.array(frames)
                frames = np.pad(
                    frames, 
                    pad_width=((0, video_start + N - i), (0, 0), (0, 0)), 
                    mode="constant",
                    constant_values=0
                )
                assert frames.shape == (N, 112, 112), "check padding"    
                return frames
        frames = np.array(frames)
        
        return frames    

    def __getitem__(self, idx, return_wave=False):
        video_path = self.video_files[idx]
        audio_clean_path = self.audio_clean_files[idx]
        audio_mix_path = self.audio_mix_files[idx] 

        x, y, start_frame = self.load_audio_pair( # x = clean, y = mix
            audio_clean_path, audio_mix_path
        )
        visual_features = self.load_video(video_path, start_frame)

        if return_wave:
            return x, y, visual_features

        X, Y = self.feature_extractor(x), self.feature_extractor(y) # [1, 48960] -> [1, 100, 192]
    
        return X, Y, torch.Tensor(visual_features) 

class MelSpecDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_ann_file,
        val_ann_file, 
        test_ann_file,
        batch_size=8,      
        num_workers=4,
        sampling_rate=24000,
        n_fft=1024, 
        hop_length=256,
        n_mels=100,
        gpu=True,
        **kwargs
    ):
        super().__init__()
    
        self.train_ann_file = train_ann_file
        self.val_ann_file = val_ann_file
        self.test_ann_file = test_ann_file
        self.batch_size = batch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.feature_extractor = MelSpectrogramFeatures(
            sample_rate=self.sampling_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        self.gpu = gpu

    def setup(self, stage=None):

        if stage == "fit" or stage is None:        
            self.train_set = MelSpec(
                ann_file=self.train_ann_file, 
                shuffle_spec=True, 
                feature_extractor=self.feature_extractor
            )
            self.valid_set = MelSpec(
                ann_file=self.val_ann_file, 
                shuffle_spec=False, 
                feature_extractor=self.feature_extractor
            )
        if stage == "test" or stage is None:
            self.test_set = MelSpec(
                ann_file=self.test_ann_file, 
                shuffle_spec=False, 
                feature_extractor=self.feature_extractor
            )            
    
    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=True,
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.valid_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False,
        )
