import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.conv7_64 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.max_pool = nn.MaxPool2d(2)

        layer = []
        for i in range(6):
            layer.append(nn.Conv2d(64, 64, 3, padding=1))
        self.conv3_64 = nn.Sequential(*layer)

        self.conv3_64_128 = nn.Conv2d(64, 128, 3, padding=1)

        layer = []
        for i in range(7):
            layer.append(nn.Conv2d(128, 128, 3, padding=1))
        self.conv3_128 = nn.Sequential(*layer)

        self.conv3_128_256 = nn.Conv2d(128, 256, 3, padding=1)

        layer = []
        for i in range(11):
            layer.append(nn.Conv2d(256, 256, 3, padding=1))
        self.conv3_256 = nn.Sequential(*layer)

        self.conv3_256_512 = nn.Conv2d(256, 512, 3, padding=1)

        layer = []
        for i in range(5):
            layer.append(nn.Conv2d(512, 512, 3, padding=1))
        self.conv3_512 = nn.Sequential(*layer)

        self.avg_pool = nn.AvgPool2d(1)

        self.fc1 = nn.Linear(512*7*7, 512)
        self.fc2 = nn.Linear(512, 120)

    def forward(self, X):
        X = self.conv7_64(X)
        X = self.max_pool(X)

        X = self.conv3_64(X)
        X = self.conv3_64_128(X)
        X = self.max_pool(X)

        X = self.conv3_128(X)
        X = self.conv3_128_256(X)
        X = self.max_pool(X)

        X = self.conv3_256(X)
        X = self.conv3_256_512(X)
        X = self.max_pool(X)

        X = self.conv3_512(X)
        X = self.avg_pool(X)

        X = X.view(X.shape[0], -1)
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        return X

net = ResNet34()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

print(net)