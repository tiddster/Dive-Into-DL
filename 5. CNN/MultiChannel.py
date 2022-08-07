import torch
from torch import nn
import DIDLutils

"""
多通道输入和输出：
我们需要构造一个输入通道数与输入数据的通道数相同的卷积核，从而能够与含多通道的输入数据做互相关运算。
"""
X = torch.tensor([
    [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ],[
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
])
print(X.shape)
K = torch.tensor([
    [
        [0, 1],
        [2, 3]
    ], [
        [1, 2],
        [3, 4]
    ]
])

print(DIDLutils.corr2d_multi_in(X, K))

K = torch.stack([K, K+1, K+2])
print(DIDLutils.corr2d_multi_in_out(X, K))

"""
1*1卷积层 ≈ 全连接层
"""
