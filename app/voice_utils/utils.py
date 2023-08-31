import numpy as np
import librosa
import copy
from scipy import signal
import torch
import soundfile as sf
import io
import base64

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

def get_spectrograms(wav):
    y, _ = librosa.effects.trim(wav, top_db=TOP_DB)
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

def _mel_to_linear_matrix(sr, n_fft, n_mels):
    m = librosa.filters.mel(sr = sr, n_fft = n_fft,n_mels = n_mels)
    m_t = np.transpose(m)
    p = np.matmul(m, m_t)
    d = [1.0 / x if np.abs(x) > 1.0e-8 else x for x in np.sum(p, axis=0)]
    return np.matmul(m_t, np.diag(d))

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

def normalize(source, target):
    mean = np.mean(np.vstack([source, target]))
    std = np.std(np.vstack([source, target]))
    source_norm = (source - mean) / std
    target_norm = (target - mean) / std
    return source_norm, target_norm, mean, std

def denormalize(mel, mean, std):
    denormalized_mel = mel * std + mean
    return denormalized_mel

def preprocess_to_torch(source_mel, target_mel, DEVICE = 'cpu'):
    source_mel = torch.from_numpy(source_mel).T.unsqueeze(0).to(DEVICE)
    target_mel = torch.from_numpy(target_mel).T.unsqueeze(0).to(DEVICE)
    return source_mel, target_mel

def convert_from_torch(torch_mel):
    mel = torch_mel.squeeze(0).numpy().T
    return mel

def byte_string_to_array(source):
    source = source.split(",")[1]
    source = base64.b64decode(source)
    source, sample_rate = sf.read(io.BytesIO(source))
    return source, sample_rate

def encode_to_byte_string(source, sample_rate):
    with io.BytesIO() as wav_file:
        sf.write(wav_file, source, sample_rate, format='wav')
        wav_file.seek(0)
        wav_data = wav_file.read()
    encoded_wav_data = base64.b64encode(wav_data).decode('utf-8')
    encoded_wav = f"data:audio/wav;base64,{encoded_wav_data}"
    return encoded_wav