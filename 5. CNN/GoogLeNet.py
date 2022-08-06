import torch
from torch import nn
import torch.nn.functional as F

import DIDLutils

"""
Inception块里有4条并行的线路。前3条线路使用窗口大小分别是1×11×1、3×33×3和5×55×5的卷积层来抽取不同空间尺寸下的信息，
其中中间2个线路会对输入先做1×11×1卷积来减少输入通道数，以降低模型复杂度。第四条线路则使用3×33×3最大池化层，后接1×11×1卷积层来改变通道数。
4条线路都使用了合适的填充来使输入与输出的高和宽一致。
最后我们将每条线路的输出在通道维上连结，并输入接下来的层中去。
"""

class Inception(nn.Module):
    def __init__(self, in_channel, out1, out2, out3, out4):
        super(Inception, self).__init__()
        self.p1_1 = nn.Conv2d(in_channel, out1, kernel_size=1)

        self.p2_1 = nn.Conv2d(in_channel, out2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(out2[0], out2[1], kernel_size=3)

        self.p3_1 = nn.Conv2d(in_channel, out3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(out3[0], out3[1], kernel_size=5)

        self.p4_1 = nn.Conv2d(in_channel, out4, kernel_size=1)

    def forward(self, X):
        p1 = F.relu(self.p1_1(X))
        p2 = F.relu(self.p2_2(self.p2_1(X)))
        p3 = F.relu(self.p3_2(self.p3_1(X)))
        p4 = F.relu(self.p4_1(X))
        return torch.cat((p1, p2, p3, p4), dim=1)


b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   DIDLutils.GlobalAvgPool2d())

net = nn.Sequential(b1, b2, b3, b4, b5,
                    DIDLutils.FlattenLayer(), nn.Linear(1024, 10))

X = torch.rand(1, 1, 96, 96)
for blk in net.children():
    X = blk(X)
    print('output shape: ', X.shape)