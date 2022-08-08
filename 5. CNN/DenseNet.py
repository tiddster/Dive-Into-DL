import torch
from torch import nn
import torch.nn.functional as F

import DIDLutils
import LoadFashionMNIST

"""
稠密链接网络
"""

def conv_block(in_channels, out_channels):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    )
    return blk

class DenseBlock(nn.Module):
    def __init__(self,num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i * out_channels
            net.append(conv_block(in_c, out_channels))
        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs * out_channels

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim = 1)
        return X

blk = nn.Sequential(
    DenseBlock(2, 4, 10)
)
X = torch.rand(4, 4, 8, 8)
print(blk)
for blk in blk.children():
    X = blk(X)
    print(X.shape)

"""
过度层：由于每个稠密块都会带来通道数的增加，使用过多则会带来过于复杂的模型。过渡层用来控制模型复杂度。它通过1×11×1卷积层来减小通道数
"""
def transition_block(in_channels, out_channels):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )
    return blk

"""
构建DenseNet模型
"""
net = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2,padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2, padding=1)
)

num_channels, growth_rate = 64,32
num_convs_in_dense_blocks = [4,4,4,4]
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    denseBlock = DenseBlock(num_convs, num_channels, growth_rate)
    net.add_module(f"DenseBlock{i}", denseBlock)
    num_channels = denseBlock.out_channels
    if i != len(num_convs_in_dense_blocks) - 1:
        net.add_module(f"transition_block{i}", transition_block(num_channels, num_channels//2))
        num_channels = num_channels // 2

net.add_module("BN", nn.BatchNorm2d(num_channels))
net.add_module("relu", nn.ReLU())
net.add_module("global_avg_pool", DIDLutils.GlobalAvgPool2d()) # GlobalAvgPool2d的输出: (Batch, num_channels, 1, 1)
net.add_module("fc", nn.Sequential(DIDLutils.FlattenLayer(), nn.Linear(num_channels, 10)))

X = torch.rand((1, 1, 96, 96))
for name, layer in net.named_children():
    print(name, ' input shape:\t', X.shape)
    X = layer(X)
    print(name, ' output shape:\t', X.shape)

batch_size = 256
train_iter, test_iter = LoadFashionMNIST.load_data_fashion_mnist(batch_size, resize=64)
lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

if __name__ == '__main__':
    DIDLutils.train_CNNet(net, train_iter, test_iter, batch_size, optimizer, DIDLutils.device, num_epochs)