from collections import OrderedDict

import torch
from torch import nn

"""
继承Module类构造模型
"""


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(784, 256)
        self.active = nn.ReLU()
        self.output = nn.Linear(256, 10)

    # 定义模型的前向计算，即如何更具输入x计算返回所需要的模型输出
    def forward(self, x):
        a = self.active(self.hidden(x))
        return self.output(a)


X = torch.rand(2, 784)
net = MLP()
print(net)
print(net(X))  # net(X) == net.forward(X)
print(net.forward(X))

"""
Module的子类: Sequential
它可以接收一个子模块的有序字典（OrderedDict）或者一系列子模块作为参数来逐一添加Module的实例
而模型的前向计算就是将这些实例按添加的顺序逐一计算
"""


# 实现一个mySequential类
class MySequential(nn.Module):
    def __init__(self, *args):
        super(MySequential, self).__init__()
        # 如果传入的是一个OrderedDict
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        # 如果传入一系列子模块
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, inputs):
        for module in self._modules.values():
            inputs = module(inputs)
        return inputs


net = MySequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
print(net)
print(net(X))

"""
ModuleList类
"""
net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
net.append(nn.Linear(256, 10)) # 类似List的append操作
# 查看层的方式--索引访问
print(net[-1])  # 类似List的索引访问
print(net)

"""
ModuleDict类
"""
net = nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'active': nn.ReLU(),
})
net['output'] = nn.Linear(256, 10)
# 查看层结构的方式一
print(net['linear'])
# 查看层结构的方式二
print(net.output)
print(net)