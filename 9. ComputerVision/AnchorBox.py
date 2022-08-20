from PIL import Image
import cv2

import numpy as np
import torch

import DIDLutils

"""
目标检测算法通常会在输入图像中采样大量的区域，然后判断这些区域中是否包含我们感兴趣的目标，并调整区域边缘从而更准确地预测目标的真实边界框（
"""
path = "F:\Dataset\catdog.png"
img = cv2.imread(path)
w, h, _ = img.shape

X = torch.Tensor(1, 3, h, w)  # 构造输入数据
Y = DIDLutils.MultiBoxPrior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
print(Y.shape) # torch.Size([1, 2042040, 4])


