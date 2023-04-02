import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                #ConvNorm(80, 512,
                ConvNorm(512, 512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(512))
        )

        for i in range(1, 5 - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(512,
                             512,
                             kernel_size=5, stride=1,
                             padding=2,
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(512))
            )

        self.convolutions.append(
            nn.Sequential(
                #ConvNorm(512, 80,
                ConvNorm(512, 512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(512))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))

        x = self.convolutions[-1](x)
        return x 

class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class DisentangledVAE(nn.Module):

    def __init__(self, speaker_size,input_sz = (1, 64, 80),
                kernel_szs = [512,512,512],
                hidden_sz: int = 256,
                latent_sz: int = 32,
                c: float = 512,
                c_delta: float = 0.001,
                beta: float = 0.1,
                beta_delta: float = 0,
                dim_neck=64, latent_dim=64, dim_pre=512, batch_size=10, device = None):
        super(DisentangledVAE, self).__init__()

        self.batch_size = batch_size
        self._input_sz = input_sz
        self._channel_szs = [input_sz[0]] + kernel_szs
        self._hidden_sz = hidden_sz
        self._c = c
        self._c_delta = c_delta
        self._beta = beta
        self._beta_delta = beta_delta
        self.latent_dim = latent_dim
        self.dim_neck = dim_neck
        self.speaker_size = speaker_size
        self.postnet = Postnet()
        self.device = device
        ############################## Encoder Architecture ###################
        self.enc_modules = []
        for i in range(3):
            conv_layer = nn.Sequential(
                #ConvNorm(80 if i==0 else 512,
                ConvNorm(512,
                        512,
                        kernel_size=5, stride=1,
                        padding=2,
                        dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(512)
            )
            self.enc_modules.append(conv_layer)
        self.enc_modules = nn.ModuleList(self.enc_modules)
        self.enc_lstm = nn.LSTM(dim_pre, dim_neck, 2, batch_first=True, bidirectional=True)

        self.enc_linear = LinearNorm(16384, 2048)

        self.style = LinearNorm(2048, self.speaker_size*2)
        self.content = LinearNorm(2048, (latent_dim - self.speaker_size)*2)
        ############################ Decoder Architecture ####################
        self.dec_pre_linear1 = nn.Linear(latent_dim, 2048)
        self.dec_pre_linear2 = nn.Linear(2048, 16384)
        self.dec_lstm1 = nn.LSTM(dim_neck*2, 512, 1, batch_first=True)
        self.dec_modules = []

        for i in range(3):
            if i==0:
                dec_conv_layer =  nn.Sequential(           
                        nn.Conv1d(dim_pre,
                                dim_pre,
                                kernel_size=5, stride=1,
                                padding=2, dilation=1),
                        nn.BatchNorm1d(dim_pre))
            else:
                dec_conv_layer =  nn.Sequential(           
                        nn.Conv1d(dim_pre,
                                dim_pre,
                                kernel_size=5, stride=1,
                                padding=2, dilation=1),
                        nn.BatchNorm1d(dim_pre))
            self.dec_modules.append(dec_conv_layer)
        self.dec_modules = nn.ModuleList(self.dec_modules)

        self.dec_lstm2 = nn.LSTM(dim_pre, 1024, 2, batch_first=True)
        self.dec_linear2 = LinearNorm(1024, 512)
        self.apply(init_weights)
    

    def encode(self, x):
        shape = x.shape
        
        for layer in self.enc_modules:
            x = F.relu(layer(x))
        x = x.transpose(1, 2)
        self.enc_lstm.flatten_parameters()

        outputs, _ = self.enc_lstm(x)
        outputs = outputs.reshape(shape[0], -1)
        outputs = F.relu(self.enc_linear(outputs))
        style = self.style(outputs)
        content = self.content(outputs)
        style_mu = style[:,:self.speaker_size]
        style_logvar = style[:,self.speaker_size:]
        content_mu = content[:,:(self.latent_dim-self.speaker_size)]
        content_logvar = content[:,(self.latent_dim-self.speaker_size):]
        return style_mu, style_logvar, content_mu, content_logvar

    def _reparameterize(self, mu, logvar, train=True):
        if train:
            epsilon = torch.autograd .Variable(torch.empty(logvar.size()).normal_()).to(self.device)
            std = logvar.mul(0.5).exp_()
            return epsilon.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        
        output = self.dec_pre_linear1(z)
        output = self.dec_pre_linear2(output)
        # print('output dims: ', output.shape)
        output = output.view(z.shape[0],-1, self.dim_neck*2)
        
        # print('-------------- decoder input shape: ', output.shape)
        output,_ = self.dec_lstm1(output)
        
        output = output.transpose(-1, -2)
        # print('-------------- output lstm shape: ', output.shape)
        for layer in self.dec_modules:
            output = F.relu(layer(output))
        output = output.transpose(-1, -2)
        # print('-------------- output lstm shape2: ', output.shape)
        output,_ = self.dec_lstm2(output)
        output = self.dec_linear2(output)
        return output.transpose(-1, -2)

    def forward(self, x1, x2, train=True):
        style_mu1, style_logvar1, content_mu1, content_logvar1 = self.encode(x1)
        z_content1 = self._reparameterize(content_mu1, content_logvar1, train)

        style_mu2, style_logvar2, content_mu2, content_logvar2 = self.encode(x2)
        z_content2 = self._reparameterize(content_mu2, content_logvar2, train)

        style_mu2 = style_mu2.detach()
        style_logvar2 = style_logvar2.detach()
        z_style_mu = (style_mu1 + style_mu2)/2
        z_style_logvar = (style_logvar1 + style_logvar2)/2
        z_style = self._reparameterize(z_style_mu, z_style_logvar)

        z1 = torch.cat((z_style, z_content1), dim=-1)
        z2 = torch.cat((z_style, z_content2), dim=-1)
        ## parameters of distribution of sample 1
        q_z1_mu = torch.cat((z_style_mu, content_mu1), dim=-1)
        q_z1_logvar = torch.cat((z_style_logvar, content_logvar1), dim=-1)

        ## parameters of distribution of sample 2
        q_z2_mu = torch.cat((z_style_mu, content_mu2), dim=-1)
        q_z2_logvar = torch.cat((z_style_logvar, content_logvar2), dim=-1)

        recons_x1 = self.decode(z1)
        recons_x2 = self.decode(z2)
        recons_x1_hat = recons_x1 + self.postnet(recons_x1)
        recons_x2_hat = recons_x2 + self.postnet(recons_x2)        
        return recons_x1, recons_x2, recons_x1_hat, recons_x2_hat,q_z1_mu, q_z1_logvar, q_z2_mu, q_z2_logvar, z_style_mu, z_style_logvar


    def update_c(self):
        self._c += self._c_delta
    
    def update_beta(self):
        self._beta += self._beta_delta