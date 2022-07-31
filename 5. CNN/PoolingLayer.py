import torch
from torch import nn

"""
池化层：为了缓解卷积层对位置的过度敏感性
池化层直接计算池化窗口内元素的最大值或者平均值。该运算也分别叫做最大池化或平均池化。
"""

"""
获得3*3的池化层，填充为1，步幅为2
"""
pool2d = nn.MaxPool2d(3, padding=1, stride=2)

X = torch.arange(16, dtype=torch.float).view((1, 1, 4, 4))
X = torch.cat((X, X + 1), dim=1)
print(pool2d(X))