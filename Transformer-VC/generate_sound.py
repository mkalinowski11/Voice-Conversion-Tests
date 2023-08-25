import librosa
import torch
import os
import numpy as np
from model import TransformerModel
from transformer_utils import (load_wave, 
                                convert_to_complex, 
                                to_decibel,
                                decibel_revert, 
                                count_parameters,
                                signal_trg_pad,
                                normalize,
                                denormalize,
                                SAMPLE_RATE
                            )
import json
import soundfile as sf

MODEL_PATH = "model_save_iter_90000.ckpt"
SOURCE_SPEECH = os.path.join("..", "data", "data", "speaker2", "arctic_a0010.wav")
TARGET_SPEECH = os.path.join("..", "data", "data", "speaker4", "arctic_a0010.wav")

CONFIG_PATH = "config.json"

CONVERSION_PATH = "conversions"
CONVERSION1_PATH = os.path.join(CONVERSION_PATH, "src_to_trg.wav")
CONVERSION2_PATH = os.path.join(CONVERSION_PATH, "trg_to_src.wav")

if __name__ == "__main__":
    print(os.getcwd())
    with open(CONFIG_PATH, "r") as file:
        config = json.load(file)
    if not os.path.exists(CONVERSION_PATH):
        os.mkdir(CONVERSION_PATH)
    # 
    model = TransformerModel(
        input_dim = config["model"]["input_dim"],
        mid_dim = config["model"]["mid_dim"],
        embed_dim = config["model"]["embed_dim"],
        num_heads = config["model"]["num_heads"],
        n_enc_blcks = config["model"]["n_enc_blcks"],
        n_dec_blcks = config["model"]["n_dec_blcks"],
        device = "cpu")
    model.load_state_dict(torch.load(MODEL_PATH))
    #
    source_amp, source_phase = load_wave(SOURCE_SPEECH)
    target_amp, target_phase = load_wave(TARGET_SPEECH)
    #
    source_amp, target_amp = to_decibel(source_amp),  to_decibel(target_amp)
    #
    norm_src, norm_trg, mean, std = normalize(source_amp, target_amp)
    src_AB, trg_AB =  norm_src, signal_trg_pad(norm_src, norm_trg)
    src_BA, trg_BA = norm_trg, signal_trg_pad(norm_trg, norm_src)
    #
    src_AB, trg_AB = torch.from_numpy(src_AB).T, torch.from_numpy(trg_AB).T
    src_BA, trg_BA = torch.from_numpy(src_BA).T, torch.from_numpy(trg_BA).T
    #
    with torch.no_grad():
        src_to_trg = model(src_AB, trg_AB).T
        trg_to_src = model(src_BA, trg_BA).T
    #
    src_to_trg, trg_to_src = denormalize(src_to_trg, trg_to_src, mean, std)
    #
    src_conversion_amp = decibel_revert(src_to_trg)
    trg_conversion_amp = decibel_revert(trg_to_src)
    #
    src_conversion = convert_to_complex(src_conversion_amp, source_phase)
    trg_conversion = convert_to_complex(trg_conversion_amp, target_phase)
    #
    src_conversion = librosa.istft(src_conversion)
    trg_conversion = librosa.istft(trg_conversion)
    #
    sf.write(CONVERSION1_PATH, src_conversion, SAMPLE_RATE)
    sf.write(CONVERSION2_PATH, trg_conversion, SAMPLE_RATE)
    print(f'Successfully converted voice, output data shape: {src_conversion_amp.shape}, {trg_conversion_amp.shape}')
    count_parameters(model)
