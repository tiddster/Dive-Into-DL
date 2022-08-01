import torch
from torch import nn, optim
import DIDLutils
import LoadFashionMNIST

"""
卷积神经网络就是含卷积层的网络
"""

device = DIDLutils.device

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2)
        )

        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84,10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], 1))
        return output


net = LeNet()
print(net)

batch_size = 256
train_iter, test_iter = LoadFashionMNIST.get_iter(batch_size)

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
DIDLutils.train_LeNet(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)