from project_utils import save_checkpoint, load_checkpoint
from generator import Generator
from discriminator import Discriminator
import torch
import torch.nn as nn
from dataset import Voice_Dataset
from torch.utils.data import DataLoader
from train_step import train_fn
import os

def main(config):
    disc_target = Discriminator(in_channels=1).to(config.DEVICE)
    disc_source = Discriminator(in_channels=1).to(config.DEVICE)
    gen_source_target = Generator(channels=1, num_residuals=9).to(config.DEVICE)
    gen_target_source = Generator(channels=1, num_residuals=9).to(config.DEVICE)

    opt_disc = torch.optim.Adam(
        list(disc_target.parameters()) + list(disc_source.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = torch.optim.Adam(
        list(gen_source_target.parameters()) + list(gen_target_source.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    if config.LOAD_MODEL:
        load_checkpoint(
            disc_target, opt_disc, config, config.CHECKPOINT_DISC_TARGET
        )
        print("tak")
        load_checkpoint(
            disc_source, opt_disc, config, config.CHECKPOINT_TRG_SRC
        )
        print("tak")
        load_checkpoint(
            gen_source_target, opt_gen, config, config.CHECKPOINT_GEN_TARGET
        )
        print("tak")
        load_checkpoint(
            gen_target_source, opt_gen, config, config.CHECKPOINT_GEN_SRC
        )
        print("tak")

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    dataset = Voice_Dataset(source_voice_path = config.SRC_VOICE_PATH,
                  target_voice_path = config.TARGET_VOICE_PATH)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.CURRENT_EPOCH, config.NUM_EPOCHS):
        print("epoch", epoch)
        train_fn(disc_target, disc_source, gen_source_target, gen_target_source, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler, config)
        if epoch % 2 == 0 and epoch != 0:
          save_checkpoint(disc_target, opt_disc, config, filename=os.path.join(config.MODELS_PATH, f'disc_target{epoch}.pth'))
          save_checkpoint(disc_source, opt_disc, config, filename=os.path.join(config.MODELS_PATH, f'disc_source{epoch}.pth'))
          save_checkpoint(gen_target_source, opt_gen, config, filename=os.path.join(config.MODELS_PATH, f'disc_gen_source{epoch}.pth'))
          save_checkpoint(gen_source_target, opt_gen, config, filename=os.path.join(config.MODELS_PATH, f'disc_gen_source{epoch}.pth'))