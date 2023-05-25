import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = num_heads)
        self.norm_1 = nn.LayerNorm((embed_dim), eps=1e-6)
        self.norm_2 = nn.LayerNorm((embed_dim), eps=1e-6)
        self.dropout_1 = nn.Dropout(p=0.1)
        self.dropout_2 = nn.Dropout(p=0.1)
        self.feed_fwd_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, x):
        attention_output, _ = self.mha(x, x, x)
        out = self.dropout_1(attention_output)
        out = self.norm_1(out + x)
        # 
        ffn_out = self.feed_fwd_net(out)
        ffn_out = self.dropout_2(ffn_out)
        out = self.norm_2(out + ffn_out)
        return out

class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads, n_enc_blcks = 9, device = "cpu"):
        super(Encoder, self).__init__()
        self.mha_list = nn.ModuleList([
            EncoderBlock(embed_dim = embed_dim, num_heads = num_heads) for _ in range(n_enc_blcks)
        ]).to(device)
    
    def forward(self, x):
        out = self.mha_list[0](x)
        for encoder_blck in self.mha_list:
            out = encoder_blck(out)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(DecoderBlock, self).__init__()
        self.mha_1 = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = num_heads)
        self.mha_2 = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = num_heads)
        self.norm_1 = nn.LayerNorm((embed_dim), eps=1e-6)
        self.norm_2 = nn.LayerNorm((embed_dim), eps=1e-6)
        self.norm_3 = nn.LayerNorm((embed_dim), eps=1e-6)
        self.dropout_1 = nn.Dropout(p=0.1)
        self.dropout_2 = nn.Dropout(p=0.1)
        self.dropout_3 = nn.Dropout(p=0.1)
        self.feed_fwd_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x, encoded_x):
        attention_output, _ = self.mha_1(x, x, x)
        out = self.dropout_1(attention_output)
        out = self.norm_1(out + x)
        # 
        out_mha2, _ = self.mha_2(encoded_x, encoded_x, x)
        out_mha2 = self.dropout_2(out_mha2)
        out = self.norm_2(out_mha2 + out)
        # 
        ffn_out = self.feed_fwd_net(out)
        ffn_out = self.dropout_3(ffn_out)
        out = self.norm_3(ffn_out + out)
        return out

class Decoder(nn.Module):
    def __init__(self, embed_dim, num_heads, n_dec_blcks = 6, device = "cpu"):
        super(Decoder, self).__init__()
        self.decoder_list = nn.ModuleList([
            DecoderBlock(embed_dim = embed_dim, num_heads = num_heads) for _ in range(n_dec_blcks)
        ]).to(device)

    def forward(self, x, encoded_x):
        for decoder_blck in self.decoder_list:
            x = decoder_blck(x, encoded_x)
        return x

class TransformerModel(nn.Module):
    def __init__(self, embed_dim, num_heads, n_enc_blcks = 9, n_dec_blcks = 9, device = "cpu"):
        super(TransformerModel, self).__init__()
        self.encoder = Encoder(embed_dim=embed_dim, num_heads=num_heads, n_enc_blcks=n_enc_blcks, device = device)
        self.decoder = Decoder(embed_dim=embed_dim, num_heads=num_heads, n_dec_blcks=n_dec_blcks, device = device)
    
    def forward(self, x_src, x_trg):
        enc_out = self.encoder(x_src)
        dec_out = self.decoder(x_trg, enc_out)
        return dec_out

class VGG_Layer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        max_pool_kernel = 2,
        max_pool_stride = (2,3),
        use_pool = False
    ):
        super(VGG_Layer, self).__init__()
        self.use_pool = use_pool
        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size = kernel_size,
                stride = stride,
                padding = padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.max_pool = nn.MaxPool2d(
            kernel_size = max_pool_kernel,
            stride = max_pool_stride
        )
    
    def forward(self, x):
        x = self.layer(x)
        if self.use_pool:
            x = self.max_pool(x)
        return x

class VGG16_ENCODER(nn.Module):
    def __init__(self, device = "cpu"):
        super(VGG16_ENCODER, self).__init__()
        self.layers1 = nn.ModuleList([VGG_Layer(1, 32, 3, 1, 1, 2, (2,3), True),
                                      VGG_Layer(1, 32, 3, 1, 1, 2, (2,3), True)]).to(device)
        self.layers2 = nn.ModuleList([VGG_Layer(32, 64, 3, 1, 1, 2, (2,3), True),
                                      VGG_Layer(32, 64, 3, 1, 1, 2, (2,3), True)]).to(device)
        self.layers3 = nn.ModuleList([VGG_Layer(64, 128, 3, 1, 1, 2, (2,3), True),
                                      VGG_Layer(64, 128, 3, 1, 1, 2, (2,3), True)]).to(device)
        self.layers4 = nn.ModuleList([VGG_Layer(128, 256, 3, 1, 1, 2, 2, True),
                                      VGG_Layer(128, 256, 3, 1, 1, 2, 2, True)]).to(device)
    
    def forward(self, x_src, y_trg):
        x1_x, x1_y = self.layers1[0](x_src), self.layers1[1](y_trg)
        # sum of extracted features branch 1 and 2
        x1_x = x1_x + x1_y
        x2_x, x2_y = self.layers2[0](x1_x), self.layers2[1](x1_y)
        # 
        x2_x = x2_x + x2_y
        x3_x, x3_y = self.layers3[0](x2_x), self.layers3[1](x2_y)
        #
        x3_x = x3_x + x3_y
        x4_x, x4_y = self.layers4[0](x3_x), self.layers4[1](x3_y)
        #
        x4_x = x4_x + x4_y
        
        return x4_x

class VGG16_DECODER(nn.Module):
    def __init__(self, device = "cpu"):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
        ).to(device)
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 5, (2,3), 1),
            nn.Conv2d(64, 64, (4,3), 1, 1),
            nn.ReLU()
        ).to(device)
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 5, (2,3), 1),
            nn.Conv2d(32, 32, (4,3), 1, 1),
            nn.ReLU()
        ).to(device)
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, 5, (2,3), 1),
            nn.Conv2d(1, 1, 4, 1, 1),
            nn.ReLU()
        ).to(device)
        
    def forward(self, x):
        out_x = self.layer1(x)
        out_x = self.layer2(out_x)
        out_x = self.layer3(out_x)
        out_x = self.layer4(out_x)
        return out_x

class PhaseModel(nn.Module):
    def __init__(self, device = "cpu"):
        super(PhaseModel, self).__init__()
        self.encoder = VGG16_ENCODER(device = device)
        self.decoder = VGG16_DECODER(device = device)
    
    def forward(self, src, trg):
        encoded_features = self.encoder(src, trg)
        decoded_features = self.decoder(encoded_features)
        return decoded_features