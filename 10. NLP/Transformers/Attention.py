import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, queries, keys, values, mask=None):
        """
        :param queries:  [batch_size, seq_len, model_dim] or [batch_size, num_head, seq_len, head_dim]
        :param keys:   [batch_size, seq_len, model_dim] or [batch_size, num_head, seq_len, head_dim]
        :param values:   [batch_size, seq_len, model_dim] or [batch_size, num_head, seq_len, head_dim]
        :return:
        """
        dk = keys.shape[-1]

        similarity = queries @ keys.transpose(-1, -2) / math.sqrt(dk)

        if mask is not None:
            similarity = similarity.masked_fill_(mask,  -1e9)
        attention_weight = F.softmax(similarity, dim=-1)
        return attention_weight @ values


class MultiHeadAttention(nn.Module):
    def __init__(self, head_dim, model_dim=512, num_heads=2, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.model_dim = model_dim
        self.num_head = num_heads
        self.head_dim = head_dim

        self.linear_Q = nn.Linear(model_dim, head_dim * num_heads)
        self.linear_K = nn.Linear(model_dim, head_dim * num_heads)
        self.linear_V = nn.Linear(model_dim, head_dim * num_heads)

        self.Attention = Attention()

        self.fwNet = nn.Sequential(
            nn.Linear(head_dim * num_heads, model_dim),
            nn.Dropout(),
        )
        self.layerNorm = nn.LayerNorm(model_dim)

    def forward(self, queries, keys, values, mask=None):
        """
        :param queries:  [batch_size, seq_len, model_dim]
        :param keys:   [batch_size, seq_len, model_dim]
        :param values:   [batch_size, seq_len, model_dim]
        :return:
        """
        res = queries
        batch_size = queries.shape[0]
        seqLen = queries.shape[1]

        # [batch_size, seq_len, head_dim * num_heads]
        queries = self.linear_Q(queries)
        keys = self.linear_K(keys)
        values = self.linear_V(values)

        queries = queries.view((batch_size, self.num_head, seqLen, self.head_dim))
        keys = keys.view((batch_size, self.num_head, seqLen, self.head_dim))
        values = values.view((batch_size,  self.num_head, seqLen, self.head_dim))

        # [batch_size, num_head, seq_len, head_dim]
        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = mask.expand((batch_size, self.num_head, mask.shape[-2], mask.shape[-1]))
        attn_outputs = self.Attention(queries, keys, values, mask=mask)
        attn_outputs = attn_outputs.view((batch_size, seqLen, self.num_head * self.head_dim))

        # [batch_size, seq_len, model_dim]
        outputs = self.fwNet(attn_outputs)

        return self.layerNorm(outputs + res)


# test_q = torch.rand((3, 10, 512))
#
# mhA = MultiHeadAttention(10)
# print(mhA(test_q, test_q, test_q).shape)