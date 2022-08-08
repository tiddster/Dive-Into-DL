import random
import time

import torch
from torch import nn
import torchvision
import torch.utils.data as UData
import torchvision.transforms as transforms
import cv2

#####################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"training on {device}")
batch_size = 256
#####################################
"""
获取数据
"""
mnist_train = torchvision.datasets.MNIST(root='F:/MINSTDataset/Num', train=True, download=True,
                                         transform=transforms.ToTensor())
mnist_test = torchvision.datasets.MNIST(root='F:/MINSTDataset/Num', train=False, download=True,
                                        transform=transforms.ToTensor())

labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
print(f"每一个图片的形状为：{mnist_train[0][0].shape}")
# 测试下载的数据是否正确
# X, Y = [], []
# for i in range(10):
#     cv2.imshow("x", mnist_train[i][0].view((28, 28)).numpy())
#     print(labels[mnist_train[i][1]])
#     cv2.waitKey(0)

train_iter = UData.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_iter = UData.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

"""
构造模型
"""


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.BatchNorm1d(120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, X):
        feature = self.conv1(X)
        feature = feature.view(X.shape[0], -1)
        label = self.fc(feature)
        return label


net = AlexNet()

"""
模型训练
"""
lr, epochs = 0.01, 2
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)


def train(net, train_iter, test_iter, batch_size, optimizer, device, epochs):
    print("training")
    net = net.to(device)
    for epoch in range(epochs):
        train_l_sum, train_right_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_l_sum += l.cpu().item()

            y_predict = y_hat.argmax(dim=1)
            train_right_sum += (y_predict == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_right_sum / n, test_acc, time.time() - start))

def evaluate_accuracy(test_iter, net, device=None):
    if device == None and isinstance(net, nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in test_iter:
            X = X.to(device)
            y = y.to(device)

            if isinstance(net, torch.nn.Module):
                net.eval()  # 评估模式
                predict_y = net(X).argmax(dim=1)
                acc_sum += (predict_y == y).float().sum().cpu().item()
                net.train()  # 训练模式
            else:
                if ('is_training' in net.__code__.co_varnames):  # 如果有is_training这个参数
                    # 将is_training设置成False
                    predict_y = net(X, is_training=False).argmax(dim=1)
                    acc_sum += (predict_y == y).float().sum().item()
                else:
                    predict_y = net(X).argmax(dim=1)
                    acc_sum += (predict_y == y).float().sum().item()
            n += y.shape[0]
            return acc_sum / n

if __name__ == "__main__":
    train(net, train_iter, test_iter, batch_size, optimizer, device, epochs)

