import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import spectral_norm


def pad_layer(inp, layer, pad_type='reflect'):
    kernel_size = layer.kernel_size[0]
    if kernel_size % 2 == 0:
        pad = (kernel_size//2, kernel_size//2 - 1)
    else:
        pad = (kernel_size//2, kernel_size//2)
    # padding
    inp = F.pad(inp, 
            pad=pad,
            mode=pad_type)
    out = layer(inp)
    return out

def pad_layer_2d(inp, layer, pad_type='reflect'):
    kernel_size = layer.kernel_size
    if kernel_size[0] % 2 == 0:
        pad_lr = [kernel_size[0]//2, kernel_size[0]//2 - 1]
    else:
        pad_lr = [kernel_size[0]//2, kernel_size[0]//2]
    if kernel_size[1] % 2 == 0:
        pad_ud = [kernel_size[1]//2, kernel_size[1]//2 - 1]
    else:
        pad_ud = [kernel_size[1]//2, kernel_size[1]//2]
    pad = tuple(pad_lr + pad_ud)
    # padding
    inp = F.pad(inp, 
            pad=pad,
            mode=pad_type)
    out = layer(inp)
    return out

def pixel_shuffle_1d(inp, scale_factor=2):
    batch_size, channels, in_width = inp.size()
    channels //= scale_factor
    out_width = in_width * scale_factor
    inp_view = inp.contiguous().view(batch_size, channels, scale_factor, in_width)
    shuffle_out = inp_view.permute(0, 1, 3, 2).contiguous()
    shuffle_out = shuffle_out.view(batch_size, channels, out_width)
    return shuffle_out

def upsample(x, scale_factor=2):
    x_up = F.interpolate(x, scale_factor=scale_factor, mode='nearest')
    return x_up

def append_cond(x, cond):
    # x = [batch_size, x_channels, length]
    # cond = [batch_size, x_channels * 2]
    p = cond.size(1) // 2
    mean, std = cond[:, :p], cond[:, p:]
    out = x * std.unsqueeze(dim=2) + mean.unsqueeze(dim=2)
    return out

def conv_bank(x, module_list, act, pad_type='reflect'):
    outs = []
    for layer in module_list:
        out = act(pad_layer(x, layer, pad_type))
        outs.append(out)
    out = torch.cat(outs + [x], dim=1)
    return out

def get_act(act):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'lrelu':
        return nn.LeakyReLU()
    else:
        return nn.ReLU()

class Inception(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Inception, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size = (1,1)), nn.ReLU())
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = (1, 1)),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = (3,3), padding = (1,1), padding_mode = "reflect"),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = (1, 1)),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = (5, 5), padding = (2, 2), padding_mode = "reflect"),
            nn.ReLU()
        )
        self.block4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.Conv2d(in_channels, out_channels, kernel_size = (1, 1), padding_mode = "reflect"),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x)
        x3 = self.block3(x)
        x4 = self.block4(x)
        return torch.cat([x1,x2, x3,x4], axis = 1)

class InceptionExtractor(nn.Module):
    def __init__(self):
        super(InceptionExtractor, self).__init__()
        self.inception1 = Inception(1, 16)# output shape [None,512, 512, 200 - dim len]
        self.inception2 = Inception(64, 32)# output shape [None, 128, 512, 200 - dim len]
        self.inception3 = Inception(128, 64)# output shape [None, 32, 512, 200 - dim len]
        self.pool1 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool3 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))
    
    def forward(self, x):
        input_shape = x.shape
        # [None, 1, 512, 200] -> [None, 1, 200, 512]
        pool1 = x.reshape(input_shape[0], input_shape[1], input_shape[3], input_shape[2])
        pool1 = self.pool1(pool1)
        # 1 inception block
        inc_1_res = self.inception1(x)
        # [None, 1, 512, 512, 200] -> [None, 1, 512, 200, 512]
        pool2 = inc_1_res.reshape(inc_1_res.shape[0], inc_1_res.shape[1], inc_1_res.shape[3], inc_1_res.shape[2])
        pool2 = self.pool2(pool2)
        # 2 inception block
        inc_2_res = self.inception2(inc_1_res)
        # [None, 128, 512, 200] -> [None, 128, 200, 512]
        pool3 = inc_2_res.reshape(inc_2_res.shape[0], inc_2_res.shape[1], inc_2_res.shape[3], inc_2_res.shape[2])
        pool3 = self.pool3(pool3)
        # 3 inception block
        inc_3_res = self.inception3(inc_2_res)
        # [None, 32, 512, 200] -> [None, 32, 200, 512]
        pool4 = inc_3_res.reshape(inc_3_res.shape[0], inc_3_res.shape[1], inc_3_res.shape[3], inc_3_res.shape[2])
        pool4 = self.pool4(pool4)
        # pool concatenation
        pool_concat = torch.cat([pool1, pool2, pool3, pool4], axis = 1).squeeze(3)
        return pool_concat, inc_3_res

class InceptionSpeakerEnc(nn.Module):
    def __init__(self):
        super(InceptionSpeakerEnc, self).__init__()
        self.inception_extractor = InceptionExtractor()
        self.conv_extractor = nn.Sequential(
            nn.MaxPool2d((4, 2)),
            nn.Conv2d(256, 128, kernel_size = (5,5)),
            nn.ReLU(),
            nn.LayerNorm((124,96)),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(128, 128, kernel_size = (3,3)),
            nn.ReLU(),
            nn.LayerNorm((60, 46)),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(128,128, kernel_size = (3,1)),
            nn.ReLU(),
            nn.LayerNorm((28, 23)),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(128,128, kernel_size = (3,3)),
            nn.ReLU(),
            nn.Conv2d(128,128, kernel_size = (4,1)),
            nn.ReLU()
        )
        self.pool_linear_extract = nn.ModuleList([
                nn.Linear(449, 200),
                nn.Linear(200, 64),
                nn.Linear(12800, 512)
            ]
        )
        self.conv_linear_extract = nn.Sequential(
            nn.Linear(10368, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )
        self.merge_linear = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )
    
    def forward(self, x):
        pool_x, inc_x = self.inception_extractor(x)
        # pool layers feature extraction
        pool_x = pool_x.reshape(pool_x.shape[0], pool_x.shape[2], pool_x.shape[1])
        for layer in self.pool_linear_extract[:-1]:
            pool_x = layer(pool_x)
            pool_x = torch.relu(pool_x)
            pool_x = F.dropout(pool_x, p = 0.3)
        pool_x = pool_x.reshape(pool_x.shape[0], -1)
        pool_x = self.pool_linear_extract[-1](pool_x)
        pool_x = torch.relu(pool_x)
        # conv layers feature extraction
        inc_x = self.conv_extractor(inc_x)
        inc_x = inc_x.reshape(inc_x.shape[0], 1, -1)
        inc_x = self.conv_linear_extract(inc_x)
        inc_x = inc_x.squeeze(1)
        # merging
        res_concat = torch.cat([pool_x, inc_x], axis = 1)
        result = self.merge_linear(res_concat)
        return result

class ContentEncoder(nn.Module):
    def __init__(self, c_in, c_h, c_out, kernel_size,
            bank_size, bank_scale, c_bank, 
            n_conv_blocks, subsample, 
            act, dropout_rate):
        super(ContentEncoder, self).__init__()
        self.n_conv_blocks = n_conv_blocks
        self.subsample = subsample
        self.act = get_act(act)
        self.conv_bank = nn.ModuleList(
                [nn.Conv1d(c_in, c_bank, kernel_size=k) for k in range(bank_scale, bank_size + 1, bank_scale)])
        in_channels = c_bank * (bank_size // bank_scale) + c_in
        self.in_conv_layer = nn.Conv1d(in_channels, c_h, kernel_size=1)
        self.first_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size) for _ \
                in range(n_conv_blocks)])
        self.second_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size, stride=sub) 
            for sub, _ in zip(subsample, range(n_conv_blocks))])
        self.norm_layer = nn.InstanceNorm1d(c_h, affine=False)
        self.mean_layer = nn.Conv1d(c_h, c_out, kernel_size=1)
        self.std_layer = nn.Conv1d(c_h, c_out, kernel_size=1)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        out = conv_bank(x, self.conv_bank, act=self.act)
        # dimension reduction layer
        out = pad_layer(out, self.in_conv_layer)
        out = self.norm_layer(out)
        out = self.act(out)
        out = self.dropout_layer(out)
        # convolution blocks
        for l in range(self.n_conv_blocks):
            y = pad_layer(out, self.first_conv_layers[l])
            y = self.norm_layer(y)
            y = self.act(y)
            y = self.dropout_layer(y)
            y = pad_layer(y, self.second_conv_layers[l])
            y = self.norm_layer(y)
            y = self.act(y)
            y = self.dropout_layer(y)
            if self.subsample[l] > 1:
                out = F.avg_pool1d(out, kernel_size=self.subsample[l], ceil_mode=True)
            out = y + out
        mu = pad_layer(out, self.mean_layer)
        log_sigma = pad_layer(out, self.std_layer)
        return mu, log_sigma

class Decoder(nn.Module):
    def __init__(self, 
            c_in, c_cond, c_h, c_out, 
            kernel_size,
            n_conv_blocks, upsample, act, sn, dropout_rate):
        super(Decoder, self).__init__()
        self.n_conv_blocks = n_conv_blocks
        self.upsample = upsample
        self.act = get_act(act)
        f = spectral_norm if sn else lambda x: x
        self.in_conv_layer = f(nn.Conv1d(c_in, c_h, kernel_size=1))
        self.first_conv_layers = nn.ModuleList([f(nn.Conv1d(c_h, c_h, kernel_size=kernel_size)) for _ \
                in range(n_conv_blocks)])
        self.second_conv_layers = nn.ModuleList(\
                [f(nn.Conv1d(c_h, c_h * up, kernel_size=kernel_size)) \
                for _, up in zip(range(n_conv_blocks), self.upsample)])
        self.norm_layer = nn.InstanceNorm1d(c_h, affine=False)
        self.conv_affine_layers = nn.ModuleList(
                [f(nn.Linear(c_cond, c_h * 2)) for _ in range(n_conv_blocks*2)])
        self.out_conv_layer = f(nn.Conv1d(c_h, c_out, kernel_size=1))
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, z, cond):
        out = pad_layer(z, self.in_conv_layer)
        out = self.norm_layer(out)
        out = self.act(out)
        out = self.dropout_layer(out)
        # convolution blocks
        for l in range(self.n_conv_blocks):
            y = pad_layer(out, self.first_conv_layers[l])
            y = self.norm_layer(y)
            y = append_cond(y, self.conv_affine_layers[l*2](cond))
            y = self.act(y)
            y = self.dropout_layer(y)
            y = pad_layer(y, self.second_conv_layers[l])
            if self.upsample[l] > 1:
                y = pixel_shuffle_1d(y, scale_factor=self.upsample[l])
            y = self.norm_layer(y)
            y = append_cond(y, self.conv_affine_layers[l*2+1](cond))
            y = self.act(y)
            y = self.dropout_layer(y)
            if self.upsample[l] > 1:
                out = y + upsample(out, scale_factor=self.upsample[l]) 
            else:
                out = y + out
        out = pad_layer(out, self.out_conv_layer)
        return out

class AE(nn.Module):
    def __init__(self, config):
        super(AE, self).__init__()
        self.speaker_encoder = InceptionSpeakerEnc()
        self.content_encoder = ContentEncoder(**config['ContentEncoder'])
        self.decoder = Decoder(**config['Decoder'])

    def forward(self, x):
        x_embed = x if len(x.shape) == 4 else x.unsqueeze(1)
        emb = self.speaker_encoder(x_embed)
        mu, log_sigma = self.content_encoder(x)
        eps = log_sigma.new(*log_sigma.size()).normal_(0, 1)
        dec = self.decoder(mu + torch.exp(log_sigma / 2) * eps, emb)
        return mu, log_sigma, emb, dec

    def inference(self, x, x_cond):
        x_cond = x_cond if len(x_cond.shape) == 4 else x_cond.unsqueeze(1)
        emb = self.speaker_encoder(x_cond)
        mu, _ = self.content_encoder(x)
        dec = self.decoder(mu, emb)
        return dec

    def get_speaker_embeddings(self, x):
        x = x if len(x.shape) == 4 else x.unsqueeze(1)
        emb = self.speaker_encoder(x)
        return emb