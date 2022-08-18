import os

import torch
from torch import nn
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

pretrained_net = models.resnet18()
print(pretrained_net.fc)

pretrained_net.fc = nn.Linear(512, 2)
print(pretrained_net.fc)
