import torch
import torch.nn as nn

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
        self.dim_per_head = d_model / num_heads #每个注意力头的输出维度
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
        
    def forward(self, batch_size, query, key, value):
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