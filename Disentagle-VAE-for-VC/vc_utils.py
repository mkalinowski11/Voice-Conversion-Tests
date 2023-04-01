import torch

def train_step(model, optimizer, loss, data1, data2):
    optimizer.zero_grad()
    recons_x1, recons_x2, recons_x1_hat,recons_x2_hat,q_z1_mu, q_z1_logvar, q_z2_mu, q_z2_logvar, style_mu, style_logvar =\
    model(data1, data2)
    loss, recons_loss1, recons_loss2, recons_loss1_hat, recons_loss2_hat, z1_kl_loss, z2_kl_loss, z_style_kl = \
    loss(data1,data2,recons_x1, recons_x2, recons_x1_hat, recons_x2_hat,q_z1_mu,q_z1_logvar,q_z2_mu, q_z2_logvar,style_mu, style_logvar)
    loss.backward()
    optimizer.step()
    return loss.item(), recons_loss1.item(), recons_loss2.item(), recons_loss1_hat.item(), recons_loss2_hat.item(), z1_kl_loss.item(), z2_kl_loss.item(), z_style_kl.item()

def loss_functionGVAE2(x1, x2, x_recon1, x_recon2, recons_x1_hat, recons_x2_hat,
                     q_z1_mu, q_z1_logvar, q_z2_mu, q_z2_logvar, style_mu1, style_logvar1, config):
        with torch.autograd.set_detect_anomaly(True):
            MSE_x1 = torch.nn.functional.l1_loss(x1, x_recon1, reduction='sum').div(config["batch_size"])
            MSE_x2 = torch.nn.functional.l1_loss(x2, x_recon2, reduction='sum').div(config["batch_size"])

            MSE_x1_hat = torch.nn.functional.l1_loss(x1, recons_x1_hat, reduction='sum').div(config["batch_size"])
            MSE_x2_hat = torch.nn.functional.l1_loss(x2, recons_x2_hat, reduction='sum').div(config["batch_size"])

            z1_kl_loss = (-0.5)*torch.sum(1 + q_z1_logvar - q_z1_mu.pow(2) - q_z1_logvar.exp(), axis=-1).mean()
            z2_kl_loss = (-0.5)*torch.sum(1 + q_z2_logvar - q_z2_mu.pow(2) - q_z2_logvar.exp(), axis=-1).mean()
            z_kl_style = (-1)*torch.sum(1 + style_logvar1 - style_mu1.pow(2) - style_logvar1.exp()).div(config["batch_size"])

            LOSS = config["mse_cof"]*(MSE_x1 + MSE_x2 + MSE_x1_hat + MSE_x2_hat) + config["kl_cof"]*(z1_kl_loss + z2_kl_loss)
        
        return LOSS, MSE_x1, MSE_x2, MSE_x1_hat, MSE_x2_hat, z1_kl_loss, z2_kl_loss

def save_model():
     pass