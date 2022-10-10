import torch
from torch import nn
from dataset import Voice_Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from discriminator import Discriminator
from generator import Generator

def load_checkpoint(model, optimizer, filename, lr, device):
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    current_epoch = checkpoint["current_epoch"]
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return current_epoch

def save_checkpoint(model, optimizer, config, filename="my_checkpoint.pth"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "current_epoch": config.CURRENT_EPOCH
    }
    torch.save(checkpoint, filename)

def train(config, source_voice_path, target_voice_path, model_data_save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Voice_Dataset(source_voice_path = source_voice_path,
                  target_voice_path = target_voice_path)
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False
    )
    gen = Generator(embed_dim=256).to(device)
    dis = Discriminator(embed_dim=256).to(device)

    gen_lr = config.optimizers['gen_lr']
    dis_lr = config.optimizers['dis_lr']
    beta1 = config.optimizers['beta1']
    beta2 = config.optimizers['beta2']

    gen_opt = torch.optim.Adam(gen.parameters(), gen_lr, [beta1, beta2])
    dis_opt = torch.optim.Adam(dis.parameters(), dis_lr, [beta1, beta2])
    
    hparam = config.hparam
    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()

    if config.load_checkpoint:
        # generator
        config.start_epoch = load_checkpoint(gen, gen_opt, config.resume_path_gen, config.optimizers['gen_lr'], device)
        # discriminator
        _ = load_checkpoint(dis, dis_opt, config.resume_path_dis, config.optimizers['dis_lr'], device)
    for epoch in range(config.start_epoch, config.num_epochs):
        loop = tqdm(train_loader, leave=True)
        for idx, (src_spect , src_embed, trg_spect , trg_embed) in enumerate(loop):
            src_spect = src_spect.to(device)
            src_embed = src_embed.to(device)
            trg_spect = trg_spect.to(device)
            trg_embed = trg_embed.to(device)
            # gen inference
            x_src_src = gen(src_spect, src_embed, src_embed)
            x_src_trg = gen(src_spect, src_embed, trg_embed)
            x_src_trg_src = gen(x_src_trg, trg_embed, src_embed)
            # discriminator
            d_src = dis(src_spect, src_embed, trg_embed)
            d_src_trg = dis(x_src_trg, trg_embed, src_embed)
            #
            dis_loss = torch.mean((d_src_trg - hparam['b']) ** 2 + (d_src - hparam['a']) ** 2)
            # reset grad discriminator
            dis_opt.zero_grad()
            dis_loss.backward(retain_graph=True)
            dis_opt.step()
            if idx % config.gen_freq == 0:
                id_loss = l2_loss(src_spect, x_src_src)
                cyc_loss = l1_loss(src_spect, x_src_trg_src)
                d_src_trg_2 = dis(x_src_trg, trg_embed, src_embed)
                adv_loss = torch.mean((d_src_trg_2 - hparam['a']) ** 2)
                gen_loss = hparam['lambda_id'] * id_loss + hparam['lambda_cyc'] * cyc_loss + adv_loss
                gen_opt.zero_grad()
                gen_loss.backward(retain_graph=True)
                gen_opt.step()
                metrics = dis_loss.item(), gen_loss.item(), adv_loss.item()
                loop.set_postfix({'dis loss':metrics[0], 'gen loss':metrics[1], 'adv loss':metrics[2]})
        if (epoch % config.epoch_save == 0) and (epoch != 0):
            save_checkpoint(gen, gen_opt, config, config.resume_path_gen)
            save_checkpoint(dis, dis_opt, config, config.resume_path_dis)
