import math

import numpy as np
import torch
from torch import nn
"""
主要实现了论文中缩放点积注意力
"""
import torch.nn.functional as F

model_dim = 512
feed_forward_dim = 1024
dk, dv = 64, 64

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()

    def forward(self, Q, K, V, mask=None):
        """
        :param Q:  [batch_size, length, feature]
        :param K:   [batch_size, KV_length, key_feature]
        :param V:   [batch_size, KV_length, value_feature]
        :return:
        """
        dk = Q.shape[-1]
        similarity = Q @ K.transpose(1, 2) / math.sqrt(dk)
        if mask:
            similarity = similarity.masked_fill_(mask, -np.inf)
        attention_weight = F.softmax(similarity, dim=1)
        print(attention_weight)
        return attention_weight @ V


# layer = ScaleDotProductAttention()
# queries = torch.normal(0, 1, (2, 3, 3))
# keys = torch.normal(0, 1, (2, 4, 3))
# values = torch.ones((2, 4, 3))
#
# output = layer(queries, keys, values)
# print(queries)
# print(keys)
# print(values)
# print(output)


class MultiHeadedAttention(nn.Module):
    def __init__(self,  dim_K, dim_V, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadedAttention, self).__init__()
        self.model_dim = model_dim
        self.num_head = num_heads
        self.dim_K = dim_K
        self.dim_V = dim_V

        self.linear_q = nn.Linear(model_dim, dim_K * num_heads)
        self.linear_k= nn.Linear(model_dim, dim_K * num_heads)
        self.linear_v = nn.Linear(model_dim, dim_V * num_heads)

        self.scale_dot_product_attention = ScaleDotProductAttention()

        self.linear_fc = nn.Linear(dim_V * num_heads, model_dim)
        self.dropout = nn.Dropout()
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, Q, K, V, mask=None):
        residual = Q

        batch_size = Q.shape[0]

        # 做一次投影运算， 将三个数据投影至每一层
        queries = self.linear_q(Q)
        keys = self.linear_k(K)
        values = self.linear_v(V)

        keys = keys.view(batch_size, -1, self.model_dim)
        queries = queries.view(batch_size, -1, self.model_dim)
        values = values.view(batch_size, -1, self.model_dim)

        if mask:
            mask = mask.unsqueeze(1).repeat(1, self.num_head, 1, 1)

        output = self.scale_dot_product_attention(queries, keys, values,  mask)
        output = output.view(batch_size, -1, self.model_dim * self.num_head)
        output = self.dropout(self.linear_fc(output))

        output = self.layer_norm(batch_size + output)

        return output

class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, model_dim)
        )

        self.layerNorm = nn.LayerNorm(model_dim)

    def forward(self, x):
        residual = x
        output = self.fc(x)
        return self.layerNorm(output + x)

# queries = torch.normal(0, 1, (2, 3, 64))
# keys = torch.normal(0, 1, (2, 4, 64))
# values = torch.ones((2, 4, 64))
# layer = MultiHeadedAttention(keys.shape[-1], values.shape[-1])
# print(layer(queries, keys, values).shape)


"""
实现mask效果，根据The Annotated Transformer提供的思路，可以生成一个上三角矩阵
"""

def padding_mask(seq_k, seq_q):
    length = seq_q.size(1)
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, length, -1)
    return pad_mask

def sequence_mask(seq):
    batch_size, length = seq.shape
    mask = torch.triu(torch.ones((length, length), dtype=torch.uint8), diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask

# Q = torch.rand((2, 6))
# K = torch.rand((2 ,5))
# print(padding_mask(K, Q))
# print(sequence_mask(Q))

"""
实现PositionEncoding
"""

class PositionEncoding(nn.Module):
    def __init__(self, model_dim=512, max_len=5000):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout()

        self.pos_emb = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        self.pos_emb[:, 0::2] = torch.sin(position * div_term)
        self.pos_emb[:, 1::2] = torch.cos(position * div_term)
        pos_emb = self.pos_emb.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pos_emb)

    def forward(self, x):
        x = x + self.pos_emb[:x.size(0), 1]
        return self.dropout(x)

"""
实现encoder和decoder
"""

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()

        self.mhaLayer = MultiHeadedAttention(512, 512)
        self.ff = FeedForward()

    def forward(self, x, mask=None):
        output = self.mhaLayer(x, x, x, mask)
        output = self.ff(output)
        return output
    
class Encoder(nn.Module):
    def __init__(self, vocab_size):
        super(Encoder, self).__init__()
        self.vocab_emb = nn.Embedding(vocab_size, model_dim)
        self.pos_emb = PositionEncoding()
        self.layers = nn.ModuleList([EncoderLayer for _ in range(6)])

    def forward(self, inputs):
        outputs = self.vocab_emb(inputs)
        outputs = self.pos_emb(outputs.transpose(0, 1)).transpose(0, 1)

        mask = padding_mask(inputs, inputs)
        for layer in self.layers:
            outputs = layer(outputs, mask)
        return outputs

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadedAttention(dk, dv)
        self.decoder_encoder_attention = MultiHeadedAttention(dk, dv)
        self.ff = FeedForward()

    def forward(self, decoder_input, encoder_outputs, self_mask, encoder_mask):
        outputs = self.self_attention(decoder_input, decoder_input, decoder_input, self_mask)
        outputs = self.decoder_encoder_attention(outputs, encoder_outputs, encoder_outputs, encoder_mask)
        output = self.ff(outputs)
        return output

class Decoder(nn.Module):
    def __init__(self, target_emb, model_dim = 512):
        super(Decoder, self).__init__()
        self.target_emb = nn.Embedding(target_emb, model_dim)
        self.pos_emb = PositionEncoding(model_dim)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(6)])

    def forward(self, decoder_inputs, encoder_inputs, encoder_outputs):
        decoder_outputs = self.target_emb(decoder_inputs)
        decoder_outputs = self.pos_emb(decoder_outputs.transpose(0, 1)).transpose(0, 1)
        self_pad_mask = padding_mask(decoder_inputs, decoder_inputs)
        seq_mask = sequence_mask(decoder_inputs)
        encoder_mask = padding_mask(decoder_inputs, encoder_inputs)
        for layer in self.layers:
            decoder_outputs = layer(decoder_outputs, encoder_outputs, self_pad_mask, encoder_mask)
        return decoder_outputs