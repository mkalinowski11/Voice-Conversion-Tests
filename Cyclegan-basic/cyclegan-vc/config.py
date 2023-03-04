import torch
import os

class Config:
    def __init__(self):
        # self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.DEVICE = "cpu"
        self.BATCH_SIZE = 4
        
        self.GEN_LR = 2e-4
        self.DISCR_LR = 1e-4
        self.GEN_LR_DECAY = self.GEN_LR / 200000
        self.DISCR_LR_DECAY = self.DISCR_LR / 200000
        self.START_DECAY = 10000
        
        self.LAMBDA_CYCLE = 10.0
        self.LAMBDA_IDENTITY = 5.0
        
        self.NUM_WORKERS = 0
        self.NUM_EPOCHS = 1000
        self.CURRENT_EPOCH = 0
        self.LOAD_MODEL = False
        self.CHECKPOINT_GEN_SRC_TRG = None
        self.CHECKPOINT_GEN_TRG_SRC = None
        self.CHECKPOINT_DISC_TRG = None
        self.CHECKPOINT_DISC_SRC = None
        #
        self.SRC_VOICE_PATH = os.path.join("..","..", "data", "data", "speaker2")
        self.TARGET_VOICE_PATH = os.path.join("..","..", "data", "data", "speaker3")
        self.MODELS_PATH = "model-save-dir"
        self.SAVE_FREQ = 100