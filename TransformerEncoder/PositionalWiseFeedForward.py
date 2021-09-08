import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalWiseFeedForward(nn.Module):
    """PositionalWiseFeedForward"""
    def __init__(self, d_model=512, d_ffn=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        #两个一维卷积核
        self.w1 = nn.Conv1d(d_model, d_ffn, 1)
        self.w2 = nn.Conv1d(d_model, d_ffn, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        output = x.transpose(1,2)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output)
        
        #残差连接
        output = self.layer_norm(x + output)
        return output