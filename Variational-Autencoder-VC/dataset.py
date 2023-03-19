import os
from torch.utils.data import Dataset
import numpy as np
import torch
from auto_vc_utils import get_spectrograms

class Voice_Dataset(Dataset):
    def __init__(self, dataset_path, target_mel_len = 128):
        self.target_mel_len = target_mel_len
        self.data = self.__prepare_dataset(dataset_path)
        self.data_mean = self.__get_mean()
        self.data_std = self.__get_std()
    
    def __prepare_dataset(self, dataset_path):
        voices = []
        for root, _, files in os.walk(dataset_path, topdown=False):
            for filename in files:
                voice_path = os.path.join(root, filename)
                mel = get_spectrograms(voice_path)
                voices.append(mel)
        voices = np.vstack(voices)
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
        return self.data.shape[0] // self.target_mel_len

    def __getitem__(self, index):
        mel = self.data[index*self.target_mel_len : index*self.target_mel_len + self.target_mel_len, :]
        mel = self.standarize(mel, self.data_mean, self.data_std)
        return torch.tensor(mel).to(torch.float32).T