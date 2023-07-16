import torch
import pandas as pd
from tqdm import tqdm
import json
from model import DisentangledVAE
from dataset import DisentagleVAEDataset
from vc_utils import loss_functionGVAE2,train_step
from torch.utils.data import DataLoader

CONFIG_PATH = "config.json"

def main(config):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = DisentagleVAEDataset(config["dataset_path"])
    model = DisentangledVAE(config["speaker_size"], device = DEVICE).to(DEVICE)
    model_loss = loss_functionGVAE2
    train_loader = DataLoader(dataset, batch_size = config["batch_size"], shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
   
    for epoch_idx in range(config["n_epochs"]):

        loop = tqdm(train_loader)
        train_loss, total_z_style_kl = 0, 0
        total_recons_loss1, total_recons_loss2, total_z1_kl_loss, total_z2_kl_loss = 0, 0, 0, 0
        total_recons_loss1_hat, total_recons_loss2_hat = 0,0
        
        metrics = []
        
        for batch_idx, (data1, data2) in enumerate(loop):
            data1 = data1.to(DEVICE)
            data2 = data2.to(DEVICE)
            
            loss, recons_loss1, recons_loss2, recons_loss1_hat, recons_loss2_hat, z1_kl_loss, z2_kl_loss, z_style_kl = train_step(model, optimizer, model_loss, data1, data2, config)
            train_loss += loss
            total_recons_loss1 += recons_loss1
            total_recons_loss2 += recons_loss2
            total_z1_kl_loss += z1_kl_loss
            total_z2_kl_loss += z2_kl_loss
            total_z_style_kl += z_style_kl
            total_recons_loss1_hat += recons_loss1_hat
            total_recons_loss2_hat += recons_loss2_hat
            # saving metrics
            metrics.append(
                (
                    loss,
                    total_recons_loss1,
                    total_recons_loss2,
                    total_z1_kl_loss,
                    total_z2_kl_loss
                )
            )
            loop.set_postfix(
                epoch = epoch_idx + 1,
                train_loss = train_loss, total_recons_loss1 = total_recons_loss1, total_recons_loss2 = total_recons_loss2,
                total_z1_kl_loss = total_z1_kl_loss, total_z2_kl_loss = total_z2_kl_loss,
                total_z_style_kl = total_z_style_kl
            )
        if (epoch_idx + 1) % config["save_freq"] or (epoch_idx + 1) == config["n_epochs"]:
            torch.save(model.state_dict(), f"model_save_epoch{epoch_idx + 1}.pth")
            torch.save(optimizer.state_dict(), f"optimizer_save_epoch{epoch_idx + 1}.opt")
            dataframe = pd.DataFrame(metrics, columns = [
                "loss",
                "total_recons_loss1",
                "total_recons_loss2",
                "total_z1_kl_loss",
                "total_z2_kl_loss"
            ])
            dataframe.to_csv(f"metrics_on_epoch_{epoch_idx + 1}.csv")

if __name__ == '__main__':
    with open(CONFIG_PATH, "r") as file:
        config = json.load(file)
    main(config)