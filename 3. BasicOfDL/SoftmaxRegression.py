from collections import OrderedDict

import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys

sys.path.append("..")
import LoadFashionMNIST as lf
import DIDutils as did

# 数据读取
train_iter, test_iter = lf.get_iter()

num_inputs = 784
num_outputs = 10
num_epochs = 5

# 构建网络
net = nn.Sequential(
    OrderedDict([
        ('flattern', did.FlattenLayer()),
        ('linear', nn.Linear(num_inputs, num_outputs))
    ])
)

# 初始化参数
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

# softmax和交叉熵损失函数
loss = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
did.trainSoftmax(net, train_iter, test_iter, loss, num_epochs, lf.batch_size, optimizer = optimizer)