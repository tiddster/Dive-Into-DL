import torchvision
from torch.utils.data import DataLoader

import DIDLutils

all_images = torchvision.datasets.CIFAR10(train=True, root="F:\Dataset\\CIFAR", download=True)

print(all_images[0][0])

def load_cifar10(is_train, augs, batch_size, root="F:\Dataset\\CIFAR"):
    dataset = torchvision.datasets.CIFAR10(root=root, train=is_train, transform=augs, download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=1)