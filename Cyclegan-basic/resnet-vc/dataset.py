import os
from torch.utils.data import Dataset
import numpy as np
import random
import torch
from project_utils import get_spectrograms, add_padding

class Voice_Dataset(Dataset):
    def __init__(self, source_voice_path, target_voice_path, target_mel_height = 200):
        self.source_voice_path = source_voice_path
        self.target_voice_path = target_voice_path
        self.target_mel_height = target_mel_height

        self.source_voices = os.listdir(self.source_voice_path)
        self.target_voices = os.listdir(self.target_voice_path)

        self.source_voices_len = len(self.source_voices)
        self.target_voices_len = len(self.target_voices)
        self.dataset_length = min(self.source_voices_len, self.target_voices_len)
        
        src_mels, trg_mels, mean, std = self.__prepare_dataset()
        self.src_mels = src_mels
        self.trg_mels = trg_mels
        self.dataset_mean = mean
        self.dataset_std = std
        self.ids = list(range(self.dataset_length))
        random.shuffle(self.ids)
    
    def __prepare_dataset(self):
        src_mels = []
        trg_mels = []
        for voice in self.source_voices:
            mel = get_spectrograms(os.path.join(self.source_voice_path, voice))
            mel = add_padding(mel, self.target_mel_height)
            src_mels.append(mel)
        for voice in self.target_voices:
            mel = get_spectrograms(os.path.join(self.target_voice_path, voice))
            mel = add_padding(mel, self.target_mel_height)
            trg_mels.append(mel)
        src_mels, trg_mels = np.array(src_mels), np.array(trg_mels)
        stacked_mels = np.vstack((src_mels, trg_mels))
        mean = np.mean(stacked_mels)
        std = np.std(stacked_mels)
        # Normalization
        src_mels = (src_mels - mean) / std
        trg_mels = (trg_mels - mean) / std
        return src_mels, trg_mels, mean, std
        
    def shuffle_ids(self):
        random.shuffle(self.ids)
    
    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index):
        sequence_idx = self.ids[index]
        return torch.tensor(self.src_mels[sequence_idx]).unsqueeze(0), torch.tensor(self.trg_mels[sequence_idx]).unsqueeze(0)