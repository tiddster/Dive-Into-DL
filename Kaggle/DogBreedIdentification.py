import os

import PIL
import cv2
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import dataloader
from torchvision.transforms import transforms

"""
读取信息, 数据处理
大致步骤：
1、 读取数据集，将csv文件中的label和filename分开， 初始化idx2label数组和label2idx字典
2、 将训练集拆分成训练集和交叉验证集（集中是图片的filename），将idx通过label2idx代替label，与filename一一对应
3、 自定义数据集
"""
# 读取数据集，将csv文件中的label和filename分开， 初始化idx2label数组和label2idx字典
CSV_PATH = "F:\Dataset\Dogs\labels.csv"
TRAIN_PATH = "F:\Dataset\Dogs\\train"
TEST_PATH = "F:\Dataset\Dogs\\test"

df = pd.read_csv(CSV_PATH)

breeds = list(df['breed'])
img_file_name = df['id']

idx2label = []
for breed in set(breeds):
    idx2label.append(breed)
label2idx = {label: idx for idx, label in enumerate(idx2label)}

# 将训练集拆分成训练集和交叉验证集（集中是图片的filename），将idx通过label2idx代替label，与filename一一对应
img_file_name = [os.path.join(TRAIN_PATH, id + ".jpg") for id in img_file_name]

img_train_path = img_file_name[:8000]
img_cross_path = img_file_name[8000:]

labels_idx = []
for i in range(len(img_file_name)):
    labels_idx.append(label2idx[breeds[i]])

labels_idx_train = labels_idx[:8000]
labels_idx_cross = labels_idx[8000:]

preprocess = transforms.Compose([
    transforms.Resize(256),
    # 将图像中央的高和宽均为224的正方形区域裁剪出来
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 自定义数据集
class LoadDataset(Dataset):
    def __init__(self, dataset_path_list):
        self.images = dataset_path_list
        self.labels = labels_idx_train

        def loader(path):
            img = PIL.Image.open(path)
            img = preprocess(img)
            return img
        self.loader = loader

    def __getitem__(self, index):
        img_path = self.images[index]
        img = self.loader(img_path)
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.images)

train_data = LoadDataset(img_train_path)
cross_data = LoadDataset(img_cross_path)

train_iter = dataloader.DataLoader(train_data, batch_size=128, shuffle=True)
cross_iter = dataloader.DataLoader(cross_data, batch_size=128, shuffle=False)

img_test_path = [os.path.join(TEST_PATH + name) for name in os.listdir(TEST_PATH)]
test_data = LoadDataset(img_test_path)
test_iter = dataloader.DataLoader(test_data, batch_size=128, shuffle=False)

