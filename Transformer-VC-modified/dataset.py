import torch
from torch.utils.data import Dataset
import os
import numpy as np
from transformer_utils import load_wave, to_decibel

class VoiceTransformerDataset(Dataset):
    def __init__(self, dataset_path, target_stft_len = 128):
        self.target_stft_len = target_stft_len
        self.data = self.__prepare_dataset(dataset_path)
        self.data_mean = self.__get_mean()
        self.data_std = self.__get_std()
        print(f"finished data preparation, dataset len: {len(self)}")
    
    def __prepare_dataset(self, dataset_path):
        voices = []
        for root, _, files in os.walk(dataset_path, topdown=False):
            for filename in files:
                voice_path = os.path.join(root, filename)
                if voice_path.endswith(".wav"):
                    amp, _ = load_wave(voice_path)
                    amp = to_decibel(amp)
                    voices.append(amp)
        voices = np.hstack(voices)
        return voices
    
    def __get_mean(self):
        mean = np.mean(self.data)
        return mean
    
    def __get_std(self):
        std = np.std(self.data)
        return std
    
    def standarize(self, X, mean, std):
        X = (X - mean) / std
        return X
    
    def __len__(self):
        return self.data.shape[1] // self.target_stft_len

    def __getitem__(self, index):
        stft = self.data[:, index*self.target_stft_len : index*self.target_stft_len + self.target_stft_len]
        stft = self.standarize(stft, self.data_mean, self.data_std)
        return torch.tensor(stft).to(torch.float32).T