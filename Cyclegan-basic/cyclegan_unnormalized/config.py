import torch
import os

class Config:
    def __init__(self):
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.BATCH_SIZE = 1
        self.LEARNING_RATE = 1e-5
        self.LAMBDA_IDENTITY = 10.0
        self.LAMBDA_CYCLE = 10
        self.NUM_WORKERS = 0
        self.NUM_EPOCHS = 1000
        self.CURRENT_EPOCH = 0
        self.LOAD_MODEL = False
        self.CHECKPOINT_GEN_TARGET = None
        self.CHECKPOINT_GEN_SRC = None
        self.CHECKPOINT_DISC_TARGET = None
        self.CHECKPOINT_TRG_SRC = None
        #
        self.SRC_VOICE_PATH = os.path.join("..", "..", "..", "..", "data", "voices_unzip", "speaker2")
        self.TARGET_VOICE_PATH = os.path.join("..", "..", "..", "..", "data", "voices_unzip", "speaker3")
        self.MODELS_PATH = "models"