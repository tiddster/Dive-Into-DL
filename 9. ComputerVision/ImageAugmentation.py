import time

import torch
from torchvision import models

import DIDLutils
import torchvision

import LoadCIFAR
from d2l import torch as d2l

img = LoadCIFAR.all_images[0][0]

def applyAug(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    DIDLutils.show_image(Y, num_rows, num_cols, scale)

# 水平翻转f
applyAug(img, torchvision.transforms.RandomHorizontalFlip())

# 竖直翻转
applyAug(img, torchvision.transforms.RandomVerticalFlip())

# 随机裁剪
shape_aug = torchvision.transforms.RandomResizedCrop(200, scale=(0.1, 1), ratio=(0.5, 2))
applyAug(img, shape_aug)

# 调整明暗度
applyAug(img, torchvision.transforms.ColorJitter(brightness=0.5))

# 改变色调
applyAug(img, torchvision.transforms.ColorJitter(hue=0.5))

# 改变对比度
applyAug(img, torchvision.transforms.ColorJitter(contrast=0.5))

color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
applyAug(img, color_aug)


# 叠加使用
augs = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
applyAug(img, augs)

def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, n, start = 0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_loss_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()

            n += y.shape[0]
            batch_count += 1
        test_acc = DIDLutils.evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_loss_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


def train_with_data_aug(train_augs, test_augs, device, lr=0.001):
    batch_size, net = 256, models.resnet18()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss()
    train_iter = LoadCIFAR.load_cifar10(True, train_augs, batch_size)
    test_iter = LoadCIFAR.load_cifar10(False, test_augs, batch_size)
    train(train_iter, test_iter, net, loss, optimizer, device, num_epochs=10)

flip_aug = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()])

no_aug = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()])

if __name__ == "__main__":
    train_with_data_aug(flip_aug, no_aug, device=DIDLutils.device)