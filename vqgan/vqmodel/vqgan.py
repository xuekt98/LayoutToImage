import importlib
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from vqgan.vqmodel.quantize import VectorQuantizer2 as VectorQuantizer
from vqgan.vqmodel.model import Encoder, Decoder, MLPBlock


class VQModel(nn.Module):
    def __init__(self,
                 edconfig,
                 n_embed,
                 embed_dim,
                 AvgMappingConfig,
                 VarMappingConfig,
                 SampleMappingConfig,
                 remap=None,
                 sane_index_shape=False):
        super(VQModel, self).__init__()
        self.encoder = Encoder(**edconfig)
        self.decoder = Decoder(**edconfig)
        self.avg_mapping = MLPBlock(**AvgMappingConfig) # 映射到均值
        self.var_mapping = MLPBlock(**VarMappingConfig) # 映射到方差
        self.avg_quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25, remap=remap, sane_index_shape=sane_index_shape)
        self.var_quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25, remap=remap, sane_index_shape=sane_index_shape)
        self.sample_mapping = MLPBlock(**SampleMappingConfig) # 从量化的结果映射回Decoder输入
        #self.quant_conv = nn.Conv2d(edconfig['z_channels'], embed_dim, 1)
        #self.post_quan_conv = nn.Conv2d(embed_dim, edconfig['z_channels'], 1)
        #self.loss = VQLPIPSWithDiscriminator(**lossconfig)

    def encode(self, x):
        # encoder input shape: (batch_size, channel=3, height, width)
        h = self.encoder(x)
        # before mapping shape: (batch_size, z_channels=256, height, width)
        #print(h.shape)
        h = h.permute(0, 2, 3, 1).to(memory_format=torch.contiguous_format)
        #print(h.shape)
        avg = self.avg_mapping(h)
        var = self.var_mapping(h)
        # after mapping shape: (batch_size, embed_dim=256, height, width)
        avg = avg.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        var = var.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        pdb.set_trace()
        avg_quant, avg_emb_loss, avg_info = self.avg_quantize(avg)
        var_quant, var_emb_loss, var_info = self.var_quantize(var)
        # after quantize shape:
        #avg_quant = avg_quant.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        #var_quant = var_quant.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        noise = torch.randn(avg_quant.shape).to(torch.device('cuda:0'))
        quant = noise * avg_quant + var_quant
        return quant, avg_emb_loss, var_emb_loss

    def decode(self, sample):
        # quant shape: (batch_size, embed_dim=256, height, width)
        # quant = self.post_quan_conv(quant)
        sample = sample.permute(0, 2, 3, 1).to(memory_format=torch.contiguous_format)
        sample = self.sample_mapping(sample)
        # decoder input shape: (batch_size, z_channels, height, width)
        sample = sample.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        dec = self.decoder(sample)
        # decoder output shape: (batch_size, channels=3, height, width)
        return dec
    
    def forward(self, input):
        sample, avg_emb_loss, var_emb_loss = self.encode(input)
        dec = self.decode(sample)
        return dec, avg_emb_loss, var_emb_loss

def instantiate_from_config(config):
    if not 'target' in config:
        raise KeyError("Expected key 'target' to instantiate")
    return get_obj_from_str(config['target'])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.split(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)