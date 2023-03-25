
import torch
import torch.nn as nn
from model import AE
from dataset import Voice_Dataset
from torch.utils.data import DataLoader
from auto_vc_utils import infinite_iter, save_model
from train_step import train_step
import json
from tqdm import tqdm
import pandas as pd

CONFIG_PATH = "config.json"

def main(config):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = AE(config).to(DEVICE)
    train_metrics = []
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["optimizer"]['lr'],
        betas=(config["optimizer"]['beta1'], config["optimizer"]['beta2']), 
        amsgrad=config["optimizer"]['amsgrad'],
        weight_decay=config["optimizer"]['weight_decay']
    )
    criterion = nn.L1Loss()
    dataset = Voice_Dataset(config["dataset_path"])
    dataloader = DataLoader(dataset, batch_size = config["data_loader"]["batch_size"], shuffle=False)
    dataloader = infinite_iter(dataloader)

    n_iterations = config["n_iterations"]
    pbar = tqdm(range(n_iterations))
    
    for iteration in pbar:
        if iteration >= config['annealing_iters']:
            lambda_kl = config['lambda']['lambda_kl']
        else:
            lambda_kl = config['lambda']['lambda_kl'] * (iteration + 1) / config['annealing_iters'] 
        data = next(dataloader)
        data = data.to(DEVICE)
        meta = train_step(model, optimizer, data, criterion, lambda_kl, config)
        loss_rec = meta['loss_rec']
        loss_kl = meta['loss_kl']
        train_metrics.append((loss_rec, loss_kl))
        pbar.set_description(f"AE: {iteration + 1}/{n_iterations}, loss_rec={loss_rec:.2f}, 'f'loss_kl={loss_kl:.2f}, lambda={lambda_kl:.1e}")

        if (iteration + 1) % config["save_frequency"] == 0 or (iteration + 1) == n_iterations:
            save_model(model, optimizer, iteration)
            dataframe = pd.DataFrame(train_metrics, columns=["loss_rec", "loss_kl"])
            dataframe.to_csv(f"metrics_on_epoch_{iteration + 1}.csv")

if __name__ == "__main__":
    with open(CONFIG_PATH, "r") as file:
        config = json.load(file)
    main(config)