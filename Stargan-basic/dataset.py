from torch.utils.data import Dataset
import os
import torch
import random

class Voice_Dataset(Dataset):
    def __init__(self, source_voice_path, target_voice_path):
        self.source_voice_spects = os.path.join(source_voice_path, "speaker1", "spects")
        self.source_voice_embeds = os.path.join(source_voice_path, "speaker1", "embeddings")
        self.target_voice_spects = os.path.join(target_voice_path, "speaker2", "spects")
        self.target_voice_embeds = os.path.join(target_voice_path, "speaker2", "embeddings")
        #
        self.source_voices_len = len(os.listdir(self.source_voice_spects))
        self.target_voices_len = len(os.listdir(self.target_voice_spects))
        #
        self.dataset_length = min(self.source_voices_len, self.target_voices_len)
        self.idxs = [*range(self.dataset_length)]
        random.shuffle(self.idxs)
    
    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index):
      src_voice_spect = torch.load(
          os.path.join(self.source_voice_spects, f"spect{index}.pth")
      )
      src_voice_embed = torch.load(
          os.path.join(self.source_voice_embeds, f"embed{index}.pth")
      )
      trg_voice_spect = torch.load(
          os.path.join(self.target_voice_spects, f"spect{index}.pth")
      )
      trg_voice_embed = torch.load(
          os.path.join(self.target_voice_embeds, f"embed{index}.pth")
      )
      return src_voice_spect, src_voice_embed, trg_voice_spect, trg_voice_embed
