from project_utils import save_checkpoint, load_checkpoint
from generator import Generator
from discriminator import Discriminator
import torch
from dataset import Voice_Dataset
from torch.utils.data import DataLoader
from train_step import train_step
import os
from config import Config
import pandas as pd

def main(config):
    generator_src_trg = Generator(channels=1, num_residuals=9).to(config.DEVICE)
    generator_trg_src = Generator(channels=1, num_residuals=9).to(config.DEVICE)
    disc_src = Discriminator().to(config.DEVICE)
    disc_trg = Discriminator().to(config.DEVICE)
    
    metrics = []

    gen_optimizer = torch.optim.Adam(
        list(generator_src_trg.parameters()) + list(generator_trg_src.parameters()),
        lr=config.GEN_LR,
        betas=(0.5, 0.999),
    )
    discr_optimizer = torch.optim.Adam(
        list(disc_src.parameters()) + list(disc_trg.parameters()),
        lr=config.DISCR_LR,
        betas=(0.5, 0.999),
    )
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_SRC_TRG, generator_src_trg, gen_optimizer, config.GEN_LR
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_TRG_SRC, generator_trg_src, gen_optimizer, config.GEN_LR
        )
        #
        load_checkpoint(
            config.CHECKPOINT_DISC_SRC, disc_src, discr_optimizer, config.DISCR_LR
        )
        load_checkpoint(
            config.CHECKPOINT_DISC_TRG, disc_trg, discr_optimizer, config.DISCR_LR
        )
    dataset = Voice_Dataset(
                            source_voice_path = config.SRC_VOICE_PATH,
                            target_voice_path = config.TARGET_VOICE_PATH
                            )
    loader =  DataLoader(
                        dataset,
                        batch_size=config.BATCH_SIZE,
                        shuffle=True,
                        num_workers=config.NUM_WORKERS,
                        pin_memory=True
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    # Main epochs loop
    for epoch_idx in range(config.NUM_EPOCHS):
        config.CURRENT_EPOCH = epoch_idx
        train_step(
            generator_src_trg, generator_trg_src,
            disc_src, disc_trg,
            loader,
            gen_optimizer,
            discr_optimizer,
            g_scaler,
            d_scaler,
            config,
            metrics
        )

        if epoch_idx + 1 in config.SAVE_MODEL_ON_EPOCHS:
            save_checkpoint(generator_src_trg, gen_optimizer, config, filename=os.path.join(config.MODELS_PATH, f'gen_src_trg{epoch_idx}.pth'))
            save_checkpoint(generator_trg_src, gen_optimizer, config, filename=os.path.join(config.MODELS_PATH, f'gen_trg_src{epoch_idx}.pth'))
            dataframe = pd.DataFrame(metrics, columns = [
                "Gen_loss",
                "Disc_loss",
                "Gen_loss_a2b",
                "Gen_loss_b2a",
                "Identity_loss",
                "Cycle_loss",
                "Disc_loss_A",
                "Disc_loss_B"
            ])
            dataframe.to_csv(f"metrics_on_epoch_{epoch_idx + 1}.csv")

if __name__ == "__main__":
    config = Config()
    main(config)