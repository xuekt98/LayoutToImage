import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

''' 
    !!!bounding box 信息如何做嵌入还需要进一步确定 
    暂时先使用
'''
class Embeddings(nn.Module):
    '''输入label，bounding box，生成嵌入的信息'''
    def __init__(self, d_model=512, num_classes=10):
        super(Embeddings, self).__init__()
        self.class_embedding = nn.Embedding(num_classes, d_model)
        self.d_model = d_model
    
    ''' !!!这个地方还需要重写 '''
    def forward(self, classes, bbs):
        '''
            Params:
                classes: 物体类别
                bb: bounding box信息
        '''
        output = self.class_embedding(classes)
        return output

class PositionalEncoding(nn.Module):
    """位置编码部分，取bounding box的中心进行编码"""
    def __init__(self, d_model=512, max_image_size=(256, 256)):
        """
        Params:
            d_model: 模型维度，默认512
            max_image_size: 生成图像的最大尺寸，默认1024，即最大1024*1024像素的图像
        """
        super(PositionalEncoding, self).__init__()
        self.H, self.W = max_image_size
        self.d_model = d_model
    
    def forward(self, boxes):
        '''
            bb_center: B x 2
        '''
        x, y = bb_center[:, 0], bb_center[:, 1]
        pos = (x * self.W + y).unsqueeze(1)
        #print(pos.size())
        pe_fe = torch.arange(self.d_model).unsqueeze(0).repeat(x.shape[0], 1).to(x.device)
        PE = pos / torch.pow(10000, 2.0 * (pe_fe // 2) / self.d_model)
        PE[:, 0::2] = torch.sin(PE[:, 0::2])
        PE[:, 1::2] = torch.cos(PE[:, 1::2])
        return PE


class ScaledDotProductAttention(nn.Module):
    """ScaledDotProductAttention"""
    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, q, k, v, scale=None):
        """
            前向传播：
            Params：
                q: Query 维度为 (batch_size, object_num, dq)
                k: Key 维度为 (batch_size, object_num, dq)
                v: Value 维度为 (batch_size, object_num, dv)
                scale: 缩放因子
            """
        attention = torch.bmm(q, k.transpose(1, 2)) #attention 张量的维度为(batch_size, object_num, object_num)
        if scale:
            attention = attention * scale
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention


class MultiHeadSelfAttention(nn.Module):
    """MultiHeadSelfAttention"""
    def __init__(self, d_model=512, num_heads=8, dropout=0.0):
        """
            Params:
                d_model: Transformer特征维度
                num_heads: 多头注意力的头个数
                dropout: dropout的概率
            """
        super(MultiHeadSelfAttention, self).__init__()
        self.dim_per_head = d_model // num_heads #每个注意力头的输出维度
        self.num_heads = num_heads          #注意力头的个数
        
        #Query, Key, Value由三个线性层实现
        self.linear_q = nn.Linear(d_model, self.dim_per_head * num_heads)
        self.linear_k = nn.Linear(d_model, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(d_model, self.dim_per_head * num_heads)
        
        #Self-Attention
        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(d_model, d_model)
        #多头注意力之后的归一化层
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value):
        """
            这里把输入的query, key, value分成三个，主要目的是应对
            不同的Attention的实现方式输入不同的情况。
            对于最原始的Attention，这三个值是相同的。
            Params：
                query：Query
                key：Key
                value: Value
            """
    
        #残差连接
        residual = query
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)
        
        #计算Query，Key，Value
        query = self.linear_q(query)
        key = self.linear_k(key)
        value = self.linear_v(value)
        
        #改变维度
        query = key.view(batch_size * num_heads, -1, dim_per_head)
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        
        #计算注意力
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(query, key, value, scale)
        
        #将多个注意力头合并
        context = context.view(batch_size, -1, dim_per_head * num_heads)
        
        output = self.linear_final(context)
        output = self.dropout(output)
        output = self.layer_norm(residual + output)
        return output, attention


class PositionalWiseFeedForward(nn.Module):
    """PositionalWiseFeedForward"""
    def __init__(self, d_model=512, d_ffn=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        #两个线性层
        self.w1 = nn.Linear(d_model, d_ffn)
        self.w2 = nn.Linear(d_model, d_ffn)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        output = self.w2(F.relu(self.w1(x)))
        output = self.dropout(output)
        
        #残差连接
        output = self.layer_norm(x + output)
        return output


class SingleEncoderLayer(nn.Module):
    """Transformer Encoder 单独的一层"""
    def __init__(self, d_model=512, num_heads=8, d_ffn=2048, dropout=0.0):
        super(SingleEncoderLayer, self).__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(d_model, d_ffn, dropout)
        
    def forward(self, inputs):
        context, attention = self.attention(inputs, inputs, inputs)
        output = self.feed_forward(context)
        return output, attention


class TransformerEncoder(nn.Module):
    """Transformer Encoder"""
    def __init__(self,
             d_model=512,
             num_classes=2500,
             num_trans_layers=6, 
             num_heads=8,
             d_ffn=2048,
             dropout=0.0):
        super(TransformerEncoder, self).__init__()
        
        self.encoder_layers = nn.ModuleList([SingleEncoderLayer(d_model, num_heads, d_ffn, dropout) for _ in range(num_trans_layers)])
        
        #!!!这里需要根据Embeddings和PositionalEncoding的实现修改
        self.input_embedding = Embeddings(d_model=d_model, num_classes=num_classes)
        self.pos_encoding = PositionalEncoding(d_model)
    
    def forward(self, object_classes, boxes):
        #!!!这里需要根据Embeddings和PositionalEncoding的实现修改
        output = self.input_embedding(object_classes, boxes)
        output += self.pos_encoding(boxes)
        
        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output)
            attentions.append(attention)
        return output, attentions