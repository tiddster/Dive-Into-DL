import torch
from torch import nn
import torch.nn.functional as F

import DIDLutils
import LoadFashionMNIST

"""
残差网络: 减少深层网络误差
"""

device = DIDLutils.device
"""
残差网络模型
"""
net = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(DIDLutils.Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(DIDLutils.Residual(out_channels, out_channels))
    return nn.Sequential(*blk)


net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
net.add_module("resnet_block2", resnet_block(64, 128, 2))
net.add_module("resnet_block3", resnet_block(128, 256, 2))
net.add_module("resnet_block4", resnet_block(256, 512, 2))
net.add_module("global_avg_pool", DIDLutils.GlobalAvgPool2d()) # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
net.add_module("fc", nn.Sequential(DIDLutils.FlattenLayer(), nn.Linear(512, 10)))

batch_size = 256

train_iter, test_iter = LoadFashionMNIST.load_data_fashion_mnist(batch_size, resize=96)

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
DIDLutils.train_CNNet(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)