import torch
import torch.nn as nn
from model import TransformerModel
from dataset import VoiceTransformerDataset
from torch.utils.data import DataLoader
from transformer_utils import infinite_iter
import json
from tqdm import tqdm
import pandas as pd

CONFIG_PATH = "config.json"

def train_step(model, optimizer, data, l1_loss, mse_loss):
    pred = model(data, data)
    l1 = l1_loss(data, pred)
    mse = mse_loss(data, pred)
    loss = (l1 * 0.5) + (mse * 0.5)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    meta = {
        "l1_loss" : l1.item(),
        "mse_loss" : mse.item()
    }
    return meta

def main(config):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # STFT Transformer
    transformer_model_stft = TransformerModel(
        embed_dim = config["model"]["embed_dim"],
        num_heads = config["model"]["num_heads"],
        n_enc_blcks = config["model"]["n_enc_blcks"],
        n_dec_blcks = config["model"]["n_dec_blcks"],
        device = DEVICE
    )
    optimizer_stft = torch.optim.Adam(
        transformer_model_stft.parameters(), lr=config["optimizer"]['lr'],
        betas=(config["optimizer"]['beta1'], config["optimizer"]['beta2']),
        weight_decay=config["optimizer"]['weight_decay']
    )
    # Phase Transformer
    transformer_model_phase = TransformerModel(
        embed_dim = config["model"]["embed_dim"],
        num_heads = config["model"]["num_heads"],
        n_enc_blcks = config["model"]["n_enc_blcks"],
        n_dec_blcks = config["model"]["n_dec_blcks"],
        device = DEVICE
    )
    optimizer_phase = torch.optim.Adam(
        transformer_model_phase.parameters(), lr=config["optimizer"]['lr'],
        betas=(config["optimizer"]['beta1'], config["optimizer"]['beta2']),
        weight_decay=config["optimizer"]['weight_decay']
    )
    #
    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()
    dataset = VoiceTransformerDataset(config["dataset_path"])
    dataloader = DataLoader(dataset, batch_size = config["data_loader"]["batch_size"])
    dataloader = infinite_iter(dataloader)
    n_iterations = config["n_iterations"]
    
    pbar = tqdm(range(n_iterations))
    train_metrics = []
    # train loop
    for iteration in pbar:
        stft, phase = next(dataloader)
        stft = stft.to(DEVICE)
        phase = phase.to(DEVICE)
        meta_stft = train_step(transformer_model_stft, optimizer_stft, stft, l1_loss, mse_loss)
        meta_phase = train_step(transformer_model_phase, optimizer_phase, phase, l1_loss, mse_loss)
        #
        l1_loss_value = meta_stft["l1_loss"]
        mse_loss_value = meta_stft["mse_loss"]
        train_metrics.append((l1_loss_value, mse_loss_value))
        pbar.set_description(f"Transformer: {iteration + 1}/{n_iterations}, l1 loss : {l1_loss_value:.4f}, mse loss : {mse_loss_value:.4f}")
        # 
        if (iteration + 1) % config["save_frequency"] == 0 or (iteration + 1) == n_iterations:
            torch.save(transformer_model_stft.state_dict(), f'model_save_iter_{iteration + 1}_STFT.ckpt')
            torch.save(transformer_model_phase.state_dict(), f'model_save_iter_{iteration + 1}_PHASE.ckpt')
            print("Saving model ===>>>")
            dataframe = pd.DataFrame(train_metrics, columns=["l1_loss", "mse_loss"])
            dataframe.to_csv(f"metrics_on_epoch_{iteration + 1}.csv")

if __name__ == "__main__":
    with open(CONFIG_PATH, "r") as file:
        config = json.load(file)
    main(config)