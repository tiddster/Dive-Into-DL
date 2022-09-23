import time

import pandas as pd
import torch
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
读取数据集
"""
submission_path = "F:\Dataset\Kaggle\digit-recognizer\\sample_submission.csv"
train_data = pd.read_csv("F:\Dataset\Kaggle\digit-recognizer\\train.csv")
test_data = pd.read_csv("F:\Dataset\Kaggle\digit-recognizer\\test.csv")

transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

"""
自定义数据集
"""
from torch.utils.data import Dataset, DataLoader
import numpy as np

Batch_size = 128


class TrainDatasetDH(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]

        image = item[1:].values.astype(np.uint8).reshape((28, 28))
        label = item[0]

        image = self.transform(image)

        return image, label


class TestDatasetDH(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]
        image = item.values.astype(np.uint8).reshape((1, 28, 28))
        label = 0
        image = torch.from_numpy(image)
        image = image.float()
        return image, label


train_dataset = TrainDatasetDH(train_data, transforms)
test_dataset = TestDatasetDH(test_data)

train_iter = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
test_iter = DataLoader(test_dataset, batch_size=Batch_size, shuffle=False)

"""
构建模型
"""
import torch.nn as nn
import torch.optim as optim


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7 * 7 * 64, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


net = VGG16()
net.to(device)

num_epochs = 100
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

def train():
    for epoch in range(num_epochs):
        train_loss_sum, train_total, train_accurate, start = 0.0, 0, 0, time.time()
        for X, labels in train_iter:
            X, labels = X.to(device), labels.to(device)
            output = net(X)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

            train_total += labels.shape[0]
            train_accurate += (output.argmax(dim=1) == labels).sum().item()

        if train_accurate / train_total > 0.995 and train_loss_sum < 0.45:
            break
        # test_acc = test_evacuate(net)
        end = time.time()
        print(
            f"{epoch}: {train_loss_sum:.3f},  {train_accurate / train_total * 100: .3f}%, {end - start:.2f}")

labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
def predict():
    # predict for test
    with torch.no_grad():
        net.eval()
        pred_labelList = []
        id = 0
        for X, _ in test_iter:
            X = X.to(device)
            outputs = net(X)
            preds = outputs.argmax(dim=1)
            for pred in preds:
                id += 1
                pred_labelList.append([id, labels[pred]])

        out_df = pd.DataFrame(pred_labelList,
                              columns=['ImageId', 'Label'])

    return out_df.to_csv(submission_path, index=False)


train()
predict()