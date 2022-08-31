import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBLK(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super(ResNetBLK, self).__init__()
        self.out_ch = out_ch
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=1)
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
        output = self.bn(self.conv2(output))

        output = self.extra(X) + output
        return F.relu(output)

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            ResNetBLK(64, 64),
            ResNetBLK(64, 128, stride=2)
        )

        self.conv3 = nn.Sequential(
            ResNetBLK(128, 128),
            ResNetBLK(128, 256, stride=2)
        )

        self.conv4 = nn.Sequential(
            ResNetBLK(256, 256),
            ResNetBLK(256, 512, stride=2)
        )

        self.conv5 = nn.Sequential(
            ResNetBLK(512, 512)
        )

        self.fc1 = nn.Linear(512*7*7, 512)
        self.fc2 = nn.Linear(512, 120)

    def forward(self, X):
        X = self.conv1(X)
        X = self.conv2(X)
        X = self.conv3(X)
        X = self.conv4(X)
        X = self.conv5(X)

        X = X.view(X.shape[0], -1)
        X = F.relu(self.fc1(X))
        X = self.fc2(X)

        return X

net = ResNet18()
X = torch.ones((2, 3, 224, 224))
print(net(X))