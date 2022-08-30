import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBLK(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super(ResNetBLK, self).__init__()
        self.out_ch = out_ch
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=3)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, stride=1, padding=1)

        self.bn = nn.BatchNorm2d(out_ch)

        self.extra = nn.Sequential()
        if in_ch != out_ch:
            self.extra = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, X):
        output = F.relu(self.bn(self.conv1(X)))
        print(output.shape)
        output = self.bn(self.conv2(output))

        output = self.extra(X) + output
        return F.relu(output)

