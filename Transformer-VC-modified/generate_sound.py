import librosa
import torch
import os
import numpy as np
from model import TransformerModel
from transformer_utils import load_wave, convert_to_complex, to_decibel, decibel_revert, SAMPLE_RATE
import json
import soundfile as sf

MODEL_PATH = "model_save_iter_10000.ckpt"
SOURCE_SPEECH = os.path.join("..", "data", "data", "speaker2", "arctic_a0010.wav")
TARGET_SPEECH = os.path.join("..", "data", "data", "speaker3", "arctic_a0010.wav")

CONFIG_PATH = "config.json"

CONVERSION_PATH = "conversions"
CONVERSION1_PATH = os.path.join(CONVERSION_PATH, "src_to_trg.wav")
CONVERSION2_PATH = os.path.join(CONVERSION_PATH, "trg_to_src.wav")

if __name__ == "__main__":
    with open(CONFIG_PATH, "r") as file:
        config = json.load(file)
    if not os.path.exists(CONVERSION_PATH):
        os.mkdir(CONVERSION_PATH)
    model = TransformerModel(
        input_dim = config["model"]["input_dim"],
        mid_dim = config["model"]["mid_dim"],
        embed_dim = config["model"]["embed_dim"],
        num_heads = config["model"]["num_heads"],
        n_enc_blcks = config["model"]["n_enc_blcks"],
        n_dec_blcks = config["model"]["n_dec_blcks"],
        device = "cpu")
    model.load_state_dict(torch.load(MODEL_PATH))

    source_amp, source_phase = load_wave(SOURCE_SPEECH)
    target_amp, target_phase = load_wave(TARGET_SPEECH)
    #
    source_amp = to_decibel(source_amp)
    target_amp = to_decibel(target_amp)
    #
    mean = np.mean(np.hstack([source_amp, target_amp]))
    std = np.std(np.hstack([source_amp, target_amp]))

    normalized_source_amp = (source_amp - mean) / std
    normalized_target_amp = (target_amp - mean) / std

    normalized_source = torch.from_numpy(normalized_source_amp).T
    normalized_target = torch.from_numpy(normalized_target_amp).T
    # Get min size of stft's
    samples_min_len = normalized_source.shape[0] if normalized_source.shape[0] < normalized_target.shape[0] else normalized_target.shape[0]

    with torch.no_grad():
        src_to_trg = model(normalized_source[:samples_min_len, :], normalized_target[:samples_min_len, :])
        trg_to_src = model(normalized_target[:samples_min_len, :], normalized_source[:samples_min_len, :])
    
    src_conversion_amp = (src_to_trg.numpy().T * std) + mean
    trg_conversion_amp = (trg_to_src.numpy().T * std) + mean

    src_conversion_amp = decibel_revert(src_conversion_amp)
    trg_conversion_amp = decibel_revert(trg_conversion_amp)
    print(source_phase.shape, samples_min_len)
    src_conversion = convert_to_complex(src_conversion_amp, source_phase[:, :samples_min_len])
    trg_conversion = convert_to_complex(trg_conversion_amp, target_phase[:, :samples_min_len])

    src_conversion = librosa.istft(src_conversion)
    trg_conversion = librosa.istft(trg_conversion)

    sf.write(CONVERSION1_PATH, src_conversion, SAMPLE_RATE)
    sf.write(CONVERSION2_PATH, trg_conversion, SAMPLE_RATE)
