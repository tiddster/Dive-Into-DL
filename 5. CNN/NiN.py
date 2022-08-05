import torch
from torch import nn

import DIDLutils
import LoadFashionMNIST

"""
NiN使用1×1卷积层来替代全连接层，从而使空间信息能够自然传递到后面的层中去。
"""

def nin_block(in_channel, out_channel, kernel_size, stride, padding):
    blk = []
    for i in range(3):
        if i == 0:
            blk.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding))
        else:
            blk.append(nn.Conv2d(out_channel, out_channel, kernel_size=1))
        blk.append(nn.ReLU())
    return nn.Sequential(*blk)

net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, stride=4, padding=0),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(96, 256, kernel_size=5, stride=1, padding=2),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(256, 384, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Dropout(0.5),
    # 标签类别数是10
    nin_block(384, 10, kernel_size=3, stride=1, padding=1),
    DIDLutils.GlobalAvgPool2d(),
    # 将四维的输出转成二维的输出，其形状为(批量大小, 10)
    DIDLutils.FlattenLayer()
)

# X = torch.rand(1, 1, 224, 224)
# for name, blk in net.named_children():
#     X = blk(X)
#     print(name, 'output shape: ', X.shape)

batch_size = 128
train_iter, test_iter = LoadFashionMNIST.load_data_fashion_mnist(batch_size, resize=224)

lr, num_epoch = 0.01, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
DIDLutils.train_CNNet(net, train_iter, test_iter, batch_size, optimizer, DIDLutils.device, num_epoch)