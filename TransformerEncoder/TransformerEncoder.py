import torch
import torch.nn as nn
import Embeddings
import PositionalEncoding
import MultiHeadSelfAttention
import PositionalWiseFeedForward

class SingleEncoderLayer(nn.Module):
    """Transformer Encoder 单独的一层"""
    def __init__(self, d_model=512, num_heads=8, d_ffn=2048, dropout=0.0):
        super(SingleEncoderLayer, slef).__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(d_model, d_ffn, dropout)
        
    def forward(self, inputs, batch_size):
        context, attention = self.attention(batch_size, inputs, inputs, inputs)
        output = self.feed_forward(context)
        return output, attention

class TransformerEncoder(nn.Module):
    """Transformer Encoder"""
    def __init__(self, 
             num_layers=6, 
             d_model=512, 
             num_heads=8,
             d_ffn=2048,
             dropout=0.0):
        super(TransformerEncoder, self).__init__()
        
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ffn, dropout) for _ in range(num_layers)])
        
        #!!!这里需要根据Embeddings和PositionalEncoding的实现修改
        self.input_embedding = Embeddings()
        self.pos_encoding = PositionalEncoding(d_model)
    
    def forward(self, inputs, inputs_len):
        #!!!这里需要根据Embeddings和PositionalEncoding的实现修改
        output = self.seq_embedding(inputs)
        output += self.pos_encoding()
        
        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output)
            attentions.append(attention)
        return output, attentions