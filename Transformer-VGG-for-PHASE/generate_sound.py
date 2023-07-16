import librosa
import torch
import os
import numpy as np
from model import TransformerModel, PhaseModel
from transformer_utils import load_wave, convert_to_complex, to_decibel, decibel_revert, SAMPLE_RATE, add_padding
import json
import soundfile as sf

MODEL_STFT_PATH = "model_save_iter_20000_STFT.ckpt"
MODEL_PHASE_PATH = "model_save_iter_20000_PHASE.ckpt"

SOURCE_SPEECH = os.path.join("..", "data", "data", "speaker2", "arctic_a0010.wav")
TARGET_SPEECH = os.path.join("..", "data", "data", "speaker3", "arctic_a0010.wav")

CONFIG_PATH = "config.json"

CONVERSION_PATH = "conversions"
CONVERSION1_PATH = os.path.join(CONVERSION_PATH, "source_target_conversion.wav")
CONVERSION2_PATH = os.path.join(CONVERSION_PATH, "target_source_conversion.wav")
TARGET_DIM = 128

if __name__ == "__main__":
    with open(CONFIG_PATH, "r") as file:
        config = json.load(file)
    if not os.path.exists(CONVERSION_PATH):
        os.mkdir(CONVERSION_PATH)

    model_stft = TransformerModel(embed_dim = config["model"]["embed_dim"],
        num_heads = config["model"]["num_heads"],
        n_enc_blcks = config["model"]["n_enc_blcks"],
        n_dec_blcks = config["model"]["n_dec_blcks"],
        device = "cpu")
    model_stft.load_state_dict(torch.load(MODEL_STFT_PATH))

    model_phase = PhaseModel(device = "cpu")
    model_phase.load_state_dict(torch.load(MODEL_PHASE_PATH))

    source_amp, source_phase = load_wave(SOURCE_SPEECH)
    target_amp, target_phase = load_wave(TARGET_SPEECH)
    #
    source_amp = to_decibel(source_amp)
    target_amp = to_decibel(target_amp)
    mean_amp = np.mean(np.hstack([source_amp, target_amp]))
    std_amp = np.std(np.hstack([source_amp, target_amp]))
    #
    source_phase = source_phase - source_phase.min()
    target_phase = target_phase - target_phase.min()
    mean_phase = np.mean(np.hstack([source_phase, target_phase]))
    std_phase = np.mean(np.hstack([source_phase, target_phase]))
    #
    normalized_source_amp = (source_amp - mean_amp) / std_amp
    normalized_target_amp = (target_amp - mean_amp) / std_amp
    #
    normalized_source_phase = (source_phase - mean_phase) / std_phase
    normalized_target_phase = (target_phase - mean_phase) / std_phase

    normalized_source_amp = torch.from_numpy(normalized_source_amp).T
    normalized_target_amp = torch.from_numpy(normalized_target_amp).T
    normalized_source_phase = torch.from_numpy(normalized_source_phase).T
    normalized_target_phase = torch.from_numpy(normalized_target_phase).T

    normalized_source_amp = add_padding(normalized_source_amp, TARGET_DIM)

    print("Source amp shape:", normalized_source_amp.shape)
    print("Target amp shape", normalized_target_amp.shape)
    print("Source phase shape:", normalized_source_phase.shape)
    print("Target phase shape:", normalized_target_phase.shape)

    min_dim = min(list(normalized_source_amp.size())[0], list(normalized_target_amp.size())[0])
    print(min_dim)
    normalized_source_amp = add_padding(normalized_source_amp, target_dim=128)
    normalized_target_amp = add_padding(normalized_target_amp, target_dim=128)
    normalized_source_phase = add_padding(normalized_source_phase, target_dim=128).unsqueeze(0).unsqueeze(0)
    normalized_target_phase = add_padding(normalized_target_phase, target_dim=128).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        # STFT processing
        source_target_amp = model_stft(normalized_source_amp, normalized_target_amp)
        target_source_amp = model_stft(normalized_target_amp, normalized_source_amp)
        # Phase processing
        source_target_phase = model_phase(normalized_source_phase, normalized_target_phase)
        target_source_phase = model_phase(normalized_target_phase, normalized_source_phase)
    source_target_phase, target_source_phase = source_target_phase.squeeze(0).squeeze(0), target_source_phase.squeeze(0).squeeze(0)
    source_target_conversion_amp = (source_target_amp.numpy().T * std_amp) + mean_amp
    target_source_conversion_amp = (target_source_amp.numpy().T * std_amp) + mean_amp
    source_target_conversion_phase = (source_target_phase.numpy().T * std_phase) + mean_phase
    target_source_conversion_phase = (target_source_phase.numpy().T * std_phase) + mean_phase

    source_target_conversion_amp = decibel_revert(source_target_conversion_amp)
    target_source_conversion_amp = decibel_revert(target_source_conversion_amp)

    source_target_voice = convert_to_complex(source_target_conversion_amp, source_target_conversion_phase)
    target_source_voice = convert_to_complex(target_source_conversion_amp, target_source_conversion_phase)

    source_target_voice = librosa.istft(source_target_voice)
    target_source_voice = librosa.istft(target_source_voice)

    sf.write(CONVERSION1_PATH, source_target_voice, SAMPLE_RATE)
    sf.write(CONVERSION2_PATH, target_source_voice, SAMPLE_RATE)
    print("Succesfully converted voice")
