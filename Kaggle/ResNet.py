import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, padding=1):
        super(ResnetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        X = self.conv1(X)
        X = self.bn1(X)
        X = self.relu1(X)
        X = self.conv2(X)
        X = self.bn2(X)
        return F.relu(X)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.layer1 = ResnetBasicBlock(3, 64)
        self.layer2 = ResnetBasicBlock(64, 128)
        self.layer3 = ResnetBasicBlock(128, 256)

        self.poolLayer = nn.AvgPool2d(2)

        self.fc1 = nn.Linear(256 * 4 * 4, 256 * 4)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(256 * 4,  256)
        self.drop2 = nn.Dropout()
        self.fc3 = nn.Linear(256, 10)

    def forward(self, X):
        X = self.layer1(X)
        X = self.poolLayer(X)
        X = self.layer2(X)
        X = self.poolLayer(X)
        X = self.layer3(X)
        X = self.poolLayer(X)

        X = X.view(X.size()[0], -1)
        X = F.relu(self.drop1(self.fc1(X)))
        X = F.relu(self.drop2(self.fc2(X)))
        X = self.fc3(X)
        return F.relu(X)


