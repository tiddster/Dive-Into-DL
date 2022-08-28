import os

import cv2
import pandas as pd
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms

"""
读取信息, 数据处理
"""
CSV_PATH = "F:\Dataset\Dogs\labels.csv"
TRAIN_PATH = "F:\Dataset\Dogs\\train.zip"
TEST_PATH = "F:\Dataset\Dogs\\test.zip"

df = pd.read_csv(CSV_PATH)

breeds = list(df['breed'])
img_id = df['id']

# 加载标签和索引
idx2label = []
for breed in set(breeds):
    idx2label.append(breed)
label2idx = {label: idx for idx, label in enumerate(idx2label)}

# 处理图像
img_id = [os.path.join(TRAIN_PATH, id + ".jpg") for id in img_id]

img_train_path = img_id[:8000]
img_cross_path = img_id[8000:]

labels_idx = []
for i in range(len(img_id)):
    labels_idx.append(label2idx[breeds[i]])

labels_idx_train = labels_idx[:8000]
labels_idx_cross = labels_idx[8000:]

preprocess = transforms.Normalize(
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
)


class trainset(Dataset):
    def __init__(self):
        self.images = img_train_path
        self.labels = labels_idx_train

        def loader(path):
            img = cv2.imread(path)
            img = img.resize((224, 224))
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

