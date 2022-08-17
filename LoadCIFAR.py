import torchvision
import DIDLutils

all_images = torchvision.datasets.CIFAR10(train=True, root="F:\Dataset\\CIFAR", download=True)

DIDLutils.show_image([all_images[i][0] for i in range(32)], 4, 8, scale=0.8)