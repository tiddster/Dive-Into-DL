import math

import torch
from torch import  nn


"""
主要实现了论文中缩放点积注意力
"""
import torch.nn.functional as F
class DotProductAttention(nn.Module):
    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        """
        :param Q:  [batch_size, length, feature]
        :param K:   [batch_size, KV_length, key_feature]
        :param V:   [batch_size, KV_length, value_feature]
        :return:
        """
        dk = Q.shape[-1]
        score = Q @ K.transpose(1, 2) / math.sqrt(dk)
        attention_weight = F.softmax(score, dim=1)
        print(attention_weight)
        return attention_weight @ V

layer = DotProductAttention()
queries = torch.normal(0, 1, (2, 3, 3))
keys = torch.normal(0, 1, (2, 4, 3))
values = torch.ones((2, 4, 3))

output = layer(queries, keys, values)
print(queries)
print(keys)
print(values)
print(output)