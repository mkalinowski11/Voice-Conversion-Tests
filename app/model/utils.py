import json
from .model import AE
import torch

def get_config(PATH):
    with open(PATH, "r") as file:
        config = json.load(file)
    return config

def get_model(MODEL_PATH, CONFIG_PATH, DEVICE = 'cpu'):
    config = get_config(CONFIG_PATH)
    model = AE(config).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)))
    return model