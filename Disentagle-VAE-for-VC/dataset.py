import torch
from torch.utils.data import Dataset
import numpy as np
import random
from vc_utils import get_spectrograms
import os

class DisentagleVAEDataset(Dataset):
    def __init__(self, data_path, shuffle = True, target_len = 128):
        speaker_paths = []
        for root, _, files in os.walk(data_path):
            for filename in files:
                voice_path = os.path.join(root, filename)
                if voice_path.endswith(".wav"):
                    speaker_paths.append(voice_path)
        if shuffle:
            random.shuffle(speaker_paths)
        self.speaker_paths = speaker_paths
        self.target_len = target_len
        print(f"found {len(self)} wavs")
    
    def __len__(self):
        return len(self.speaker_paths) // 2
    
    def __getitem__(self, index):
        voice_path1 = self.speaker_paths[index]
        voice_path2 = self.speaker_paths[len(self) + index]
        mel1 = get_spectrograms(voice_path1)
        mel2 = get_spectrograms(voice_path2)
        
        mel1, mel2 = self.__add_padding(mel1, self.target_len), self.__add_padding(mel2, self.target_len)
        return torch.tensor(mel1).to(torch.float32).reshape(-1, self.target_len), torch.tensor(mel2).to(torch.float32).reshape(-1, self.target_len)
    
    def __add_padding(self, x, target_len = 128):
        shape = x.shape
        if shape[0] < target_len:
            padding_len = target_len - shape[0]
            padding = np.zeros((padding_len, shape[1]))
            return np.concatenate((x, padding), axis = 0)
        else:
            return x[:target_len, :]