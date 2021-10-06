import torch
import torch.nn as nn
import torch.nn.functional as F
from vqgan.vqmodel.quantize import VectorQuantizer2 as VectorQuantizer
from vqgan.vqmodel.model import Encoder, Decoder


class VQModel(nn.Module):
    def __init__(self,
                 edconfig,
                 n_embed,
                 embed_dim,
                 remap=None,
                 sane_index_shape=False):
        super(VQModel, self).__init__()
        self.encoder = Encoder(**edconfig)
        self.decoder = Decoder(**edconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25, remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = nn.Conv2d(edconfig['z_channels'], embed_dim, 1)
        self.post_quan_conv = nn.Conv2d(embed_dim, edconfig['z_channels'], 1)
        #self.loss = VQLPIPSWithDiscriminator(**lossconfig)

    def encode(self, x):
        # encoder input shape: (batch_size, channel=3, height, width)
        h = self.encoder(x)
        # before quantize shape: (batch_size, z_channels=256, height, width)
        h = self.quant_conv(h)
        # after quantize shape: (batch_size, embed_dim=256, height, width)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        # quant shape: (batch_size, embed_dim=256, height, width)
        quant = self.post_quan_conv(quant)
        # decoder input shape: (batch_size, z_channels, height, width)
        dec = self.decoder(quant)
        # decoder output shape: (batch_size, channels=3, height, width)
        return dec
    
    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff