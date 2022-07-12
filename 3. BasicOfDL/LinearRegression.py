import random

import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np

#####################################
num_inputs = 2
num_sample = 1000
weight = torch.tensor([2, -3.2]).view(-1, 1)
b = 4.2
xs = torch.randn(num_sample, num_inputs)
batch_size = 10
#####################################

ys = xs @ weight + b
print(ys)

plt.plot(xs[:, 1].numpy(), ys.numpy(), '.')
plt.show()

"""
读取数据集
"""
# 将数据集读取成张量
dataset = Data.TensorDataset(xs, ys)
# 数据集迭代器，将数据集按照batch_size的大小分若干批次，以便网络训练
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
# for X, y in data_iter:
#     print(X, y)


"""
定义模型
"""
"""
写法一：构造Net类
"""


class LinearNet(nn.Module):
    def __init__(self, num_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_feature, 1)

    def forward(self, x):
        y = self.linear(x)
        return y


net1 = LinearNet(num_inputs)
print(net1)

"""
写法二：静态加载Net类
"""
net2 = nn.Sequential()
net2.add_module('0', nn.Linear(num_inputs, 1))
print(net2)

"""
写法三：一步，静态加载Net类
"""
net3 = nn.Sequential(
    nn.Linear(num_inputs, 1)
)
print(net3)

for param in net3.parameters():
    print(param)

"""
初始化模型参数：
"""
# 用写法二、三
nn.init.normal_(net3[0].weight, mean=0, std=0.01)
nn.init.constant_(net3[0].bias, val=0)

# 用写法一：
nn.init.normal_(net1.linear.weight, mean=0, std=0.01)
nn.init.constant_(net1.linear.bias, val=0)

"""
定义损失函数
"""
loss = nn.MSELoss()

"""
定义优化算法
"""
optimizer = torch.optim.SGD(net3.parameters(), lr=0.03)
print(optimizer)

"""
训练模型
"""
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net3(X)
        loss_val = loss(output, y)
        optimizer.zero_grad()  # 每一次迭代都需要梯度清零
        loss_val.backward()
        optimizer.step()  # 迭代模型
    print(f"epoch:{epoch}, loss:{loss_val}, loss:{loss_val.item()}")

"""
获取结果
"""
dense = net3[0]
print(weight, dense.weight)
print(b, dense.bias)
pre_ys = xs @ dense.weight.view(-1, 1) + dense.bias
plt.plot(xs[:, 1].numpy(), pre_ys.detach().numpy(), '.')
plt.show()
