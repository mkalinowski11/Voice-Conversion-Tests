import os
from torch.utils.data import Dataset
import librosa
import random
import torch
from project_utils import fit_sound, preprocess_sound, amp_to_decibel
from project_utils import SAMPLE_RATE

class Voice_Dataset(Dataset):
    def __init__(self, source_voice_path, target_voice_path):
        self.source_voice_path = source_voice_path
        self.target_voice_path = target_voice_path

        self.source_voices = os.listdir(self.source_voice_path)
        self.target_voices = os.listdir(self.target_voice_path)
        self.source_voices.sort()
        self.target_voices.sort()

        self.source_voices_len = len(self.source_voices)
        self.target_voices_len = len(self.target_voices)

        self.dataset_length = min(self.source_voices_len, self.target_voices_len)
        self.ids = [idx for idx in range(self.dataset_length)]
        random.shuffle(self.ids)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index):
        sequence_idx = self.ids[index]
        src_voice = os.path.join(
                                self.source_voice_path,
                                self.source_voices[sequence_idx]
                                )
        target_voice = os.path.join(
                                  self.target_voice_path,
                                  self.target_voices[sequence_idx]
                                  )
        src_voice, _ = librosa.load(src_voice, sr = SAMPLE_RATE)
        target_voice, _ = librosa.load(target_voice, sr = SAMPLE_RATE)
        # set unified size of sound wave
        src_voice, target_voice = fit_sound(src_voice), fit_sound(target_voice)
        src_voice_amp, _ = preprocess_sound(src_voice)
        target_voice_amp, _ = preprocess_sound(target_voice)
        
        src_voice_amp = amp_to_decibel(src_voice_amp)
        target_voice_amp = amp_to_decibel(target_voice_amp)
        src_voice_amp = torch.from_numpy(src_voice_amp)
        target_voice_amp = torch.from_numpy(target_voice_amp)
        return src_voice_amp.unsqueeze(0).to(torch.float32), target_voice_amp.unsqueeze(0).to(torch.float32)