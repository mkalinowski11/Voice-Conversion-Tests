import torch
import librosa
import numpy as np

CUTOFF = 64000
SPECT_CUTOFF = 501
EPS=1e-6
RANDOM_SIGNAL_LEVEL_DB = -40.0
SAMPLE_RATE = 16000

def save_checkpoint(model, optimizer, config, filename="my_checkpoint.pth"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "current_epoch": config.CURRENT_EPOCH
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, config, filename="my_checkpoint.pth"):
  checkpoint = torch.load(filename, map_location=config.DEVICE)
  model.load_state_dict(checkpoint["state_dict"])
  optimizer.load_state_dict(checkpoint["optimizer"])
  config.CURRENT_EPOCH = checkpoint["current_epoch"]
  for param_group in optimizer.param_groups:
        param_group["lr"] = config.LEARNING_RATE

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

def normalize(spect):
    spect_mean, spect_std = spect.mean(), spect.std()
    spect_normalized = (spect - spect_mean) / spect_std
    return spect_normalized, spect_mean, spect_std

def denormalize(spect, spect_mean, spect_std):
    return spect * spect_std + spect_mean