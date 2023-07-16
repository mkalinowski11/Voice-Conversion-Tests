import librosa
import numpy as np

FRAME_SHIFT = 0.0125
FRAME_LENGTH = 0.05
SAMPLE_RATE = 16000
TOP_DB = 15
PREEMHPASIS = 0.97
N_FFT = 2048
MAX_DB = 100

def load_wave(file_path, sr=SAMPLE_RATE):
    y, sr = librosa.load(file_path, sr=sr)
    y, _ = librosa.effects.trim(y, top_db=TOP_DB)
    linear = librosa.stft(y=y,
                          n_fft=N_FFT)
    amplitude, phase = np.abs(linear), np.angle(linear)
    return amplitude, phase


def convert_to_complex(amplitude, phase):
    return amplitude * np.vectorize(complex)(np.cos(phase), np.sin(phase))

def to_decibel(S, ref = 1.0, eps = 1e-9):
    return 10 * np.log10(S / ref + eps)

def decibel_revert(db):
    return 10 ** (db / 10)

def infinite_iter(loader):
    it = iter(loader)
    while True:
        try:
            ret = next(it)
            yield ret
        except StopIteration:
            it = iter(loader)