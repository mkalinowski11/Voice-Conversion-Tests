import torch
import librosa
import numpy as np

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

def get_spectrograms(fpath):
    """
    Returns mel spect from wav file.
    """
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

def infinite_iter(loader):
    it = iter(loader)
    while True:
        try:
            ret = next(it)
            yield ret
        except StopIteration:
            it = iter(loader)

def save_model(model, optimizer, iteration):
        torch.save(model.state_dict(), f'model_save_iter_{iteration}.ckpt')
        torch.save(optimizer.state_dict(), f'opt_save_iter_{iteration}.opt')
        print("Saving model ===>>>")