import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn as nn
import time

# 先进行变换, 将数据进行组合变换，变换成tensor并且归一化
from torch import optim

transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

train_dataset = torchvision.datasets.CIFAR10(root='F:\Dataset\CIFAR', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='F:\Dataset\CIFAR', train=True, download=True, transform=transform)

train_iter = data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
test_iter = data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv7 = nn.Conv2d(256, 256, 1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()

        self.conv8 = nn.Conv2d(256, 512, 3, 1)
        self.conv9 = nn.Conv2d(512, 512, 3, 1)
        self.conv10 = nn.Conv2d(512, 512, 1, 1)
        self.pool4 = nn.MaxPool2d(2, 2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU()

        self.fc1 = nn.Linear(32*4*4, 256)
        self.drop1 = nn.Dropout2d()
        self.fc2 = nn.Linear(256, 256)
        self.drop2 = nn.Dropout2d()
        self.out= nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = x.view(-1,32 * 4 * 4)
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = self.out(x)
        return x


lr, batch_size, num_epochs = 0.001, 256, 5

net = Net()
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)


def train(net):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    for epoch in range(num_epochs):
        train_loss_sum, train_correct_sum, n, batch_count,  start = 0.0, 0.0, 0, 0,  time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            output = net(X)
            l = loss(output, y)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_loss_sum += l.item()
            train_correct_sum += (output.argmax(dim=1) == y).sum().item()
            n += output.shape[0]
            batch_count += 1
        test_acc = evacuate_accuracy(test_iter, net)
        end = time.time()
        print(f"epoch:{epoch}, train_loss_sum:{train_loss_sum}, train_acc_sum:{train_correct_sum / n}, test_acc_sum:{test_acc}, time：{end - start}")


def evacuate_accuracy(test_iter, net):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    correct, total = 0, 0
    for X, y in test_iter:
        X, y = X.to(device), y.to(device)
        output = net(X)

        correct += (output.argmax(dim=1) == y).sum().item()
        total += y.shape[0]
    return correct / total

if __name__ == '__main__':
    train(net)
