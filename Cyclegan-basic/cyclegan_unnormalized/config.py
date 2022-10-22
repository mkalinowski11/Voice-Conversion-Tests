import torch

class Config:
  def __init__(self):
    self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    self.BATCH_SIZE = 1
    self.LEARNING_RATE = 1e-5
    self.LAMBDA_IDENTITY = 10.0
    self.LAMBDA_CYCLE = 10
    self.NUM_WORKERS = 1
    self.NUM_EPOCHS = 1000
    self.CURRENT_EPOCH = 0
    self.LOAD_MODEL = True
    self.CHECKPOINT_GEN_TARGET = "/content/gdrive/MyDrive/test/gen_target10.pth"
    self.CHECKPOINT_GEN_SRC = "/content/gdrive/MyDrive/test/gen_source10.pth"
    self.CHECKPOINT_DISC_TARGET = "/content/gdrive/MyDrive/test/disc_target10.pth"
    self.CHECKPOINT_TRG_SRC = "/content/gdrive/MyDrive/test/disc_source10.pth"
    self.SRC_VOICE_PATH = "/content/gdrive/MyDrive/voice_data/voices_unzip/speaker4"
    self.TARGET_VOICE_PATH = "/content/gdrive/MyDrive/voice_data/voices_unzip/speaker3"
    self.MODELS_PATH = "/content/gdrive/MyDrive/test"