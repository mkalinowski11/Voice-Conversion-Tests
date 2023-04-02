import torch
import librosa
import numpy as np

FRAME_SHIFT = 0.0125
FRAME_LENGTH = 0.05
SAMPLE_RATE = 16000
TOP_DB = 15
PREEMHPASIS = 0.97
N_FFT = 2048
HOP_LENGTH = int(SAMPLE_RATE*FRAME_SHIFT)
WIN_LENTGH = int(SAMPLE_RATE*FRAME_LENGTH)
N_MELS = 512
REF_DB = 20
MAX_DB = 100
N_GRIFFIN_LIM_ITER = 100
FRAME_SIZE = 1

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

def get_spectrograms(fpath):
    y, sr = librosa.load(fpath, sr=SAMPLE_RATE)
    y, _ = librosa.effects.trim(y, top_db=TOP_DB)
    y = np.append(y[0], y[1:] - PREEMHPASIS * y[:-1])
    # stft
    linear = librosa.stft(y=y,
                          n_fft=N_FFT,
                          hop_length=HOP_LENGTH,
                          win_length=WIN_LENTGH)

    mag = np.abs(linear)
    # mel spectrogram
    mel_basis = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS)
    mel = np.dot(mel_basis, mag)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))

    # normalize
    mel = np.clip((mel - REF_DB + MAX_DB) / MAX_DB, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)

    return mel