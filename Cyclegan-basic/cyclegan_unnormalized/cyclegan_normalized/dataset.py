import numpy as np
import os
from torch.utils.data import Dataset
import librosa
import random
import torch

SAMPLE_RATE = 16000
CUTOFF = 64000
SPECT_CUTOFF = 501
EPS=1e-6
RANDOM_SIGNAL_LEVEL_DB = -40.0

def preprocess_sound(audio):
  audio_stft = librosa.stft(audio, n_fft=512)
  amplitude, phase = np.abs(audio_stft), np.angle(audio_stft)
  return amplitude, phase

def convert_to_complex(amplitude, phase):
  return amplitude * np.vectorize(complex)(np.cos(phase), np.sin(phase))

def amp_to_decibel(S, ref = 1.0):
  #return 10 * np.log10( (S + EPS)  / ref)
  return 10 * np.log10( (S)  / ref)

def decibel_revert(db):
  return 10 ** (db / 10)

def signal_pad(signal, fixed_length = CUTOFF, noise_level = RANDOM_SIGNAL_LEVEL_DB):
  pad_length = fixed_length - signal.shape[0]
  pad = (np.random.rand(pad_length) - 0.5) * 2 * decibel_revert(noise_level)
  sound_extended = np.concatenate((signal, pad), axis=0)
  return sound_extended

def fit_sound(wav, cutoff = CUTOFF):
  if wav.shape[0] < cutoff:
    signal_padding = signal_pad(wav, fixed_length = cutoff)
    return signal_padding
  return wav[:cutoff]

def fit_spectrogram_db(spect, cutoff = SPECT_CUTOFF):
  if spect.shape[1] >= cutoff:
    x = spect[:, :cutoff]
    return x
  else:
    while spect.shape[1] != cutoff:
      number_of_samples = spect.shape[1] if cutoff - spect.shape[1] >= spect.shape[1] else cutoff - spect.shape[1]
      padding = spect[:, :number_of_samples]
      spect = np.concatenate((spect, padding), axis = 1)
    return spect

def normalize(spect):
  spect_mean, spect_std = spect.mean(), spect.std()
  spect_normalized = (spect - spect_mean) / spect_std
  return spect_normalized, spect_mean, spect_std

def denormalize(spect, spect_mean, spect_std):
  return spect * spect_std + spect_mean

def min_max_scaling(spect):
  spect_max, spect_min = spect.max(), spect.min()
  spect_min_max = (spect - spect_min) / (spect_max - spect_min)
  return spect_min_max, spect_max, spect_min

def min_max_descale(spect, spect_max, spect_min):
  spect_descaled = spect * (spect_max - spect_min) + spect_min
  return spect_descaled

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
      src_voice, sr = librosa.load(src_voice, sr = SAMPLE_RATE)
      target_voice, sr = librosa.load(target_voice, sr = SAMPLE_RATE)
      #
      src_voice, target_voice = fit_sound(src_voice), fit_sound(target_voice)
      # phase is not required for training
      src_voice_amp, _ = preprocess_sound(src_voice)
      target_voice_amp, _ = preprocess_sound(target_voice)
      #
      src_voice_amp = amp_to_decibel(src_voice_amp)
      target_voice_amp = amp_to_decibel(target_voice_amp)
      # convert to torch
      src_voice_amp = torch.from_numpy(src_voice_amp)
      target_voice_amp = torch.from_numpy(target_voice_amp)
      return src_voice_amp.unsqueeze(0).to(torch.float32), target_voice_amp.unsqueeze(0).to(torch.float32)