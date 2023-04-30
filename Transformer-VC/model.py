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
    def __init__(self, embed_dim, num_heads, n_enc_blcks = 9):
        super(Encoder, self).__init__()
        self.mha_list = nn.ModuleList([
            EncoderBlock(embed_dim = embed_dim, num_heads = num_heads) for _ in range(n_enc_blcks)
        ])
    
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
    def __init__(self, embed_dim, num_heads, n_dec_blcks = 6):
        super(Decoder, self).__init__()
        self.decoder_list = nn.ModuleList([
            DecoderBlock(embed_dim = embed_dim, num_heads = num_heads) for _ in range(n_dec_blcks)
        ])

    def forward(self, x, encoded_x):
        for decoder_blck in self.decoder_list:
            x = decoder_blck(x, encoded_x)
        return x

class TransformerModel(nn.Module):
    def __init__(self, embed_dim, num_heads, n_enc_blcks = 9, n_dec_blcks = 9):
        super(TransformerModel, self).__init__()
        self.encoder = Encoder(embed_dim=embed_dim, num_heads=num_heads, n_enc_blcks=n_enc_blcks)
        self.decoder = Decoder(embed_dim=embed_dim, num_heads=num_heads, n_dec_blcks=n_dec_blcks)
    
    def forward(self, x_src, x_trg):
        enc_out = self.encoder(x_src)
        dec_out = self.decoder(x_trg, enc_out)
        return dec_out