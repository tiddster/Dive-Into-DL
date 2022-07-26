import torch
from torch import nn
from torch.nn import init

net = nn.Sequential(
    nn.Linear(4, 3),
    nn.ReLU(),
    nn.Linear(3, 2)
)
X = torch.rand(2, 4)
y = net(X)

print(net)
print(y)

# 访问多层感知机net的所有参数
# print(type(net.named_parameters()))
# for name, param in net.named_parameters():
#     print(name, param)

# 将权重参数初始化成均值为0、标准差为0.01的正态分布随机数，
# 将偏差参数初始化成1
for name, param in net.named_parameters():
    if 'weight' in name:
        init.normal_(param, mean=0, std=0.01)
        print(name, param.data)
    if 'bias' in name:
        init.constant_(param, val=1)
        print(name, param.data)


# init.normal_的实现方法
def normal_(tensor, mean=0, std=1):
    with torch.no_grad():
        return tensor.normal_(mean, std)

"""
多层共享参数：Module类的forward函数里多次调用同一个层。
此外，如果我们传入Sequential的模块是同一个Module实例的话参数也是共享的
"""
print("多层共享参数--------------------------")
linear = nn.Linear(2, 2, bias=False)
net = nn.Sequential(linear, linear)
print(net)
for name, param in net.named_parameters():
    init.constant_(param, val=3)
    print(name, param.data)