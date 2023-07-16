import torch
from torch.utils.data import Dataset
import os
import numpy as np
from transformer_utils import load_wave, to_decibel

class VoiceTransformerDataset(Dataset):
    def __init__(self, dataset_path, target_stft_len = 128):
        self.target_stft_len = target_stft_len
        stfts, phases = self.__prepare_dataset(dataset_path)
        self.stfts = stfts
        self.phases = phases
        mean_stft, mean_phase = self.__get_means()
        self.mean_stft = mean_stft
        self.mean_phase = mean_phase
        std_stft, std_phase = self.__get_stds()
        self.std_stft = std_stft
        self.std_phase = std_phase
        print(f"finished data preparation, dataset len: {len(self)}")
    
    def __prepare_dataset(self, dataset_path):
        stfts = []
        phases = []
        for root, _, files in os.walk(dataset_path, topdown=False):
            for filename in files:
                voice_path = os.path.join(root, filename)
                if voice_path.endswith(".wav"):
                    amp, phase = load_wave(voice_path)
                    amp = to_decibel(amp)
                    phase = phase - phase.min()
                    stfts.append(amp)
                    phases.append(phase)
        stfts = np.hstack(stfts)
        phases = np.hstack(phases)
        return stfts, phases
    
    def __get_means(self):
        mean_stft = np.mean(self.stfts)
        mean_phase = np.mean(self.phases)
        return mean_stft, mean_phase
    
    def __get_stds(self):
        std_stft = np.std(self.stfts)
        std_phase = np.std(self.phases)
        return std_stft, std_phase
    
    def standarize(self, X, mean, std):
        X = (X - mean) / std
        return X
    
    def __len__(self):
        return self.stfts.shape[1] // self.target_stft_len

    def __getitem__(self, index):
        # STFT
        stft = self.stfts[:, index*self.target_stft_len : index*self.target_stft_len + self.target_stft_len]
        stft = self.standarize(stft, self.mean_stft, self.std_stft)
        # Phase
        phase = self.phases[:, index*self.target_stft_len : index*self.target_stft_len + self.target_stft_len]
        phase = self.standarize(phase, self.mean_phase, self.std_phase)
        return torch.tensor(stft).to(torch.float32).T, torch.tensor(phase).to(torch.float32).T.unsqueeze(0)