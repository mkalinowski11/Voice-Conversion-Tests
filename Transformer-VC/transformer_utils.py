import librosa
import numpy as np
from prettytable import PrettyTable
import copy

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

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    # print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def signal_trg_pad(source, target):
    #
    # returns target signal shaped as source
    #
    src_len = source.shape[1]
    trg_len = target.shape[1]
    if src_len <= trg_len:
        return target[:, :src_len]
    else:
        padding = np.zeros((source.shape[0], src_len - trg_len), dtype=target.dtype)
        new_trg = np.concatenate([target, padding], axis = 1)
        return new_trg

def normalize(source, target):
    mean = np.mean(np.hstack([source, target]))
    std = np.std(np.hstack([source, target]))
    normalized_source = (source - mean) / std
    normalized_target = (target - mean) / std
    return normalized_source, normalized_target, mean, std

def denormalize(source, target, mean, std):
    source = (source.numpy() * std) + mean
    target = (target.numpy() * std) + mean
    return source, target
