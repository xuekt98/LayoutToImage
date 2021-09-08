import torch
import torch.nn as nn
import numpy as np
import FCBlock


class VecToMask(nn.Module):
    """利用Transformer Encoder的输出生成Mask"""
    def __init__(self, in_features=512, feature_grid_size=(8, 8, 8),
                 V2G_hidden_features=256, V2G_num_hidden_layers=3,
                 Decoder_hidden_features=256, Decoder_num_hidden_layers=3,
                 mask_image_size=(64, 64, 64)):

        self.in_features = in_features
        self.feature_grid_size = feature_grid_size
        self.V2G_hidden_features = V2G_hidden_features
        self.V2G_num_hidden_layers = V2G_num_hidden_layers
        self.Decoder_hidden_features = Decoder_hidden_features
        self.Decoder_num_hidden_layers = Decoder_num_hidden_layers
        self.mask_image_size = mask_image_size
        
        #从TransformerEncoder输出的vector生成图像网格
        self.V2G_net = FCBlock(in_features=in_features, out_features=np.prod(feature_grid_size),
                               hidden_features=V2G_hidden_features, num_hidden_layers=V2G_num_hidden_layers,
                               outermost_linear=True, nonlinearity='relu')

        self.Decoder_net = FCBlock(in_features=, out_features=np.prod(mask_image_size),
                                   hidden_features=Decoder_hidden_features, num_hidden_layers=Decoder_num_hidden_layers,
                                   outermost_linear=True, nonlinearity='relu')

    def forward(self, model_input):
        output = self.V2G_net(model_input)
        return output
