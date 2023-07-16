import librosa
import torch
import os
import numpy as np
from model import AE
from auto_vc_utils import get_spectrograms, melspectrogram2wav, SAMPLE_RATE
import json
import soundfile as sf

MODEL_PATH = "model_save_iter_99999.ckpt"
SOURCE_SPEECH = os.path.join("..", "data", "data", "speaker2", "arctic_a0010.wav")
TARGET_SPEECH = os.path.join("..", "data", "data", "speaker3", "arctic_a0010.wav")

CONFIG_PATH = "config.json"

CONVERSION_PATH = "conversions"
CONVERSION1_PATH = os.path.join(CONVERSION_PATH, "male_to_fem.wav")
CONVERSION2_PATH = os.path.join(CONVERSION_PATH, "fem_to_male.wav")

if __name__ == "__main__":
    with open(CONFIG_PATH, "r") as file:
        config = json.load(file)
    if not os.path.exists(CONVERSION_PATH):
        os.mkdir(CONVERSION_PATH)
    model = AE(config)
    model.load_state_dict(torch.load(MODEL_PATH))
    source_wav = get_spectrograms(SOURCE_SPEECH)
    target_wav = get_spectrograms(TARGET_SPEECH)

    mean = np.mean(np.vstack([source_wav, target_wav]))
    std = np.std(np.vstack([source_wav, target_wav]))

    normalized_source = (source_wav - mean) / std
    normalized_target = (target_wav - mean) / std

    normalized_source = torch.from_numpy(normalized_source[:200, :])
    normalized_target = torch.from_numpy(normalized_target[:200, :])

    with torch.no_grad():
        male_to_fem = model.inference(normalized_source.T.unsqueeze(0), normalized_target.T.unsqueeze(0))
        fem_to_male = model.inference(normalized_target.T.unsqueeze(0), normalized_source.T.unsqueeze(0))
    
    male_to_fem = (male_to_fem.squeeze(0).numpy().T * std) + mean
    fem_to_male = (fem_to_male.squeeze(0).numpy().T * std) + mean

    male_to_fem = melspectrogram2wav(male_to_fem)
    fem_to_male = melspectrogram2wav(fem_to_male)

    sf.write(CONVERSION1_PATH, male_to_fem, SAMPLE_RATE)
    sf.write(CONVERSION2_PATH, fem_to_male, SAMPLE_RATE)