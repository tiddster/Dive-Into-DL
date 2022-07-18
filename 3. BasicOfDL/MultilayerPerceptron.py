import torch
from torch.nn import init

import LoadFashionMNIST as lfm
import DIDLutils as didl_utils
import numpy as np
import torch.nn as nn

batch_size = lfm.batch_size
train_iter, test_iter = lfm.get_iter()

num_inputs, num_outputs, num_hiddens = 784, 10, 256

w1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
b1 = torch.zeros(num_hiddens, dtype=torch.float)
w2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)

params = [w1, b1, w2, b2]
for param in params:
    param.requires_grad_(requires_grad=True)

# 从0开始实现
# """
#  模型
# """
# def net(x):
#     x = x.view((-1, num_inputs))
#     H = didl_utils.relu(torch.matmul(x, w1) + b1)
#     O = torch.matmul(H, w2) + b2
#     return O
#
#
# loss = torch.nn.CrossEntropyLoss()
#
# num_epochs, lr = 5, 0.001
# didl_utils.train(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)

net = nn.Sequential(
    didl_utils.FlattenLayer(),
    nn.Linear(num_inputs, num_hiddens),
    nn.Sigmoid(),
    nn.Linear(num_hiddens, num_outputs)
)

for param in net.parameters():
    init.normal_(param, mean=0, std=0.01)

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

loss = torch.nn.CrossEntropyLoss()

num_epochs, lr = 5, 0.001
didl_utils.train(net, train_iter, test_iter, loss, num_epochs, batch_size, optimizer=optimizer)
