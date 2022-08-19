import os

import torch
from d2l.torch import d2l
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torchvision import models

import DIDLutils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = "F:\Dataset\hotdog\hotdog"


# 读取数据
train_imgs = ImageFolder(os.path.join(path, 'train'))
test_imgs = ImageFolder(os.path.join(path), 'test')

print(train_imgs)

hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
DIDLutils.show_image(hotdogs + not_hotdogs, 2, 8, scale=1.4)

normalize_aug = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_augs = transforms.Compose([
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_aug
])
test_augs = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    normalize_aug
])

# 查看残差网络的输出层与模型不匹配
pretrained_net = models.resnet18()
print(pretrained_net.fc)

pretrained_net.fc = nn.Linear(512, 2)
print(pretrained_net.fc)

output_params = list(map(id, pretrained_net.fc.parameters()))
feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())

lr = 0.01
optimizer = optim.SGD([
    {
        'params': feature_params
    }, {
        'params': pretrained_net.fc.parameters(),
        'lr': lr * 10
    }
],  lr=lr, weight_decay=0.001)


# 利用微调训练
def train_fine_tuning(net, optimizer, batch_size=128, num_epochs=5):
    train_iter = DataLoader(ImageFolder(os.path.join(path, 'train'), transform=train_augs),
                            batch_size, shuffle=True)
    test_iter = DataLoader(ImageFolder(os.path.join(path, 'test'), transform=test_augs),
                           batch_size)
    loss = torch.nn.CrossEntropyLoss()
    DIDLutils.train(net, train_iter, test_iter, loss, num_epochs, batch_size, optimizer=optimizer)


train_fine_tuning(pretrained_net, optimizer)


# 利用随机初始化的参数训练
scratch_net = models.resnet18(pretrained=False, num_classes=2)
lr = 0.1
optimizer = optim.SGD(scratch_net.parameters(), weight_decay=0.001)
train_fine_tuning(scratch_net, optimizer)