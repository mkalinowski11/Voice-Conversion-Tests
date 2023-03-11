import torch
import librosa
import numpy as np
import copy
from scipy import signal
import torch.nn.functional as F

SAMPLE_RATE = 16000
FRAME_SHIFT = 0.0125
FRAME_LENGTH = 0.05
TOP_DB = 15
PREEMHPASIS = 0.97
N_FFT = 2048
HOP_LENGTH = int(SAMPLE_RATE*FRAME_SHIFT)
WIN_LENTGH = int(SAMPLE_RATE*FRAME_LENGTH)
N_MELS = 512
REF_DB = 20
MAX_DB = 100
N_GRIFFIN_LIM_ITER = 100
FRAME_SIZE = 1

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

def _mel_to_linear_matrix(sr, n_fft, n_mels):
    m = librosa.filters.mel(sr = sr, n_fft = n_fft,n_mels = n_mels)
    m_t = np.transpose(m)
    p = np.matmul(m, m_t)
    d = [1.0 / x if np.abs(x) > 1.0e-8 else x for x in np.sum(p, axis=0)]
    return np.matmul(m_t, np.diag(d))

def invert_spectrogram(spectrogram):
    '''
    spectrogram: [f, t]
    '''
    return librosa.istft(spectrogram, hop_length = HOP_LENGTH, win_length=WIN_LENTGH, window="hann")

def griffin_lim(spectrogram):
    X_best = copy.deepcopy(spectrogram)
    for i in range(N_GRIFFIN_LIM_ITER):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, n_fft = N_FFT, hop_length = HOP_LENGTH, win_length = WIN_LENTGH)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)
    return y

def get_spectrograms(fpath):
    y, sr = librosa.load(fpath, sr=SAMPLE_RATE)
    y, _ = librosa.effects.trim(y, top_db=TOP_DB)
    y = np.append(y[0], y[1:] - PREEMHPASIS * y[:-1])
    # stft
    linear = librosa.stft(y=y,
                          n_fft=N_FFT,
                          hop_length=HOP_LENGTH,
                          win_length=WIN_LENTGH)

    mag = np.abs(linear)
    # mel spectrogram
    mel_basis = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS)
    mel = np.dot(mel_basis, mag)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))

    # normalize
    mel = np.clip((mel - REF_DB + MAX_DB) / MAX_DB, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)

    return mel

def melspectrogram2wav(mel):
    mel = mel.T
    # de-noramlize
    mel = (np.clip(mel, 0, 1) * MAX_DB) - MAX_DB + REF_DB

    # to amplitude
    mel = np.power(10.0, mel * 0.05)
    m = _mel_to_linear_matrix(SAMPLE_RATE, N_FFT, N_MELS)
    mag = np.dot(m, mel)
    # wav reconstruction
    wav = griffin_lim(mag)
    # de-preemphasis
    wav = signal.lfilter([1], [1, -PREEMHPASIS], wav)
    # trim
    wav, _ = librosa.effects.trim(wav)
    return wav.astype(np.float32)

def wave_feature_extraction(wav_file, sr):
    y, sr = librosa.load(wav_file, sr)
    y, _ = librosa.effects.trim(y, top_db=20)
    return y

def spec_feature_extraction(wav_file):
    mel = get_spectrograms(wav_file)
    return mel

def utt_make_frames(x, frame_size = FRAME_SIZE):
    remains = x.size(0) % frame_size
    if remains != 0:
        x = F.pad(x, (0, remains))
    out = x.view(1, x.size(0) // frame_size, frame_size * x.size(1)).transpose(1, 2)
    return out

def normalize(x, mean, std):
    ret = (x - mean) / std
    return ret

def add_padding(x, expected_size = 256):
    if x.shape[0] >= expected_size:
        return x[:expected_size, :]
    else:
        padding_len = expected_size - x.shape[0]
        padding = np.zeros((padding_len, x.shape[1]))
        return np.concatenate((x, padding), axis = 0)