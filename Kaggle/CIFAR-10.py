import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn as nn
import time
import ResNet18

# 先进行变换, 将数据进行组合变换，变换成tensor并且归一化
from torch import optim

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
    ]
)

train_dataset = torchvision.datasets.CIFAR10(root='F:\Dataset\CIFAR', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='F:\Dataset\CIFAR', train=False, download=True, transform=transform)

train_iter = data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
test_iter = data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


import torch.nn.functional as F

class Net(nn.Module):
    # 把网络中具有可学习参数的层放在构造函数__inif__中
    def __init__(self):
        # 下式等价于nn.Module.__init__.(self)
        super(Net, self).__init__()  # RGB 3*32*32
        self.conv1 = nn.Conv2d(3, 15, 3)  # 输入3通道，输出15通道，卷积核为3*3
        self.conv2 = nn.Conv2d(15, 75, 4)  # 输入15通道，输出75通道，卷积核为4*4
        self.conv3 = nn.Conv2d(75, 375, 3)  # 输入75通道，输出375通道，卷积核为3*3
        self.fc1 = nn.Linear(1500, 400)  # 输入2000，输出400
        self.fc2 = nn.Linear(400, 120)  # 输入400，输出120
        self.fc3 = nn.Linear(120, 84)  # 输入120，输出84
        self.fc4 = nn.Linear(84, 10)  # 输入 84，输出 10（分10类）

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # 3*32*32  -> 150*30*30  -> 15*15*15
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 15*15*15 -> 75*12*12  -> 75*6*6
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)  # 75*6*6   -> 375*4*4   -> 375*2*2
        print(x.shape)
        x = x.view(x.size()[0], -1)  # 将375*2*2的tensor打平成1维，1500
        x = F.relu(self.fc1(x))  # 全连接层 1500 -> 400
        x = F.relu(self.fc2(x))  # 全连接层 400 -> 120
        x = F.relu(self.fc3(x))  # 全连接层 120 -> 84
        x = self.fc4(x)  # 全连接层 84  -> 10
        return x

lr, num_epochs = 0.001, 10

net = ResNet18.ResNet()
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

from torch.autograd import Variable
def train(net):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    for epoch in range(num_epochs):  # 训练8次
        running_loss, n, correct, start = 0.0, 0, 0, time.time()
        for i, (X,y) in enumerate(train_iter):
            # enumerate() 函数：用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中
            # enumerate(sequence, [start=0])
            # sequence -- 一个序列、迭代器或其他支持迭代对象。
            # start -- 下标起始位置。

            # 输入数据
            X, y = X.to(device), y.to(device)
            X, y = Variable(X), Variable(y)

            # 梯度清零
            optimizer.zero_grad()

            # forward + backward
            outputs = net(X)
            l = loss(outputs, y)  # 计算单个batch误差
            l.backward()  # 反向传播

            # 更新参数
            optimizer.step()

            # 打印log信息
            running_loss += l.item()  # 2000个batch的误差和
            correct += (outputs.argmax(dim=1) == y).sum().item()
            n += y.shape[0]

            if i % 2000 == 1999:  # 每2000个batch打印一次训练状态
                print("[%d,%5d] loss: %.3f" \
                      % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        end = time.time()
        test_acc = evacuate_accuracy(test_iter, net)
        print(f"test_acc:{test_acc}, train_acc:{correct/n}, time: %.2f" % (end-start))
    print("Finished Training")


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
