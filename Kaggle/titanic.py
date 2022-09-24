import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import math

"""
读取数据并进行各种数据预处理
"""
train = pd.read_csv("F:\Dataset\Kaggle\\titanic\\train.csv")
test = pd.read_csv("F:\Dataset\Kaggle\\titanic\\test.csv")
submission = pd.read_csv("F:\Dataset\Kaggle\\titanic\\gender_submission.csv")

train = train.drop('PassengerId', axis=1)
test = test.drop('PassengerId', axis=1)
print(train.head(3))


# 为第二行名字更换一个显著特征
def change_name(name):
    if "Mr." in name:
        return "Mr"
    elif "Mrs." in name:
        return "Mrs"
    elif "Miss." in name:
        return "Miss"
    elif "Master." in name:
        return "Master"
    elif "Ms.":
        return "Ms"
    else:
        return "No"


train['Name'] = train['Name'].apply(change_name)
test['Name'] = test['Name'].apply(change_name)

"""
姓名特征
"""
median_value = dict(train.groupby('Name')['Age'].median())
train['Age'] = train.apply(lambda x: int(median_value[x.Name]) if math.isnan(x.Age) else int(x.Age), axis=1)
test['Age'] = test.apply(lambda x: int(median_value[x.Name]) if math.isnan(x.Age) else int(x.Age), axis=1)
print(train.head(3))

train['Fare'] = train['Fare'].apply(lambda x: 100 if x >= 100 else x)
test['Fare'] = test['Fare'].apply(lambda x: 100 if x >= 100 else x)

train = train.drop('Ticket', axis=1)
test = test.drop('Ticket', axis=1)

train = train.drop("Cabin", axis=1)
test = test.drop("Cabin", axis=1)

"""
将一个种类中的多个变量用one-hot编码表示
"""
train = pd.get_dummies(train, columns=['Pclass', 'Name', 'Sex', 'Embarked'], drop_first=True)
test = pd.get_dummies(test, columns=['Pclass', 'Name', 'Sex', 'Embarked'], drop_first=True)

y = train['Survived']
X = train.drop('Survived', axis=1)

"""
数据归一化
"""
import torchvision.transforms as transforms

"""
构建数据集
"""
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
from sklearn.preprocessing import MinMaxScaler

y = train['Survived']
X = train.drop('Survived',axis=1)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
test = scaler.transform(test)
X = torch.tensor(X,dtype=torch.float32)
y = torch.tensor(y,dtype = torch.float32)
test = torch.tensor(test,dtype=torch.float32)
y = y.reshape(-1,1)

batch_size = 1
train_dataset = TensorDataset(X, y)
train_iter = DataLoader(train_dataset, batch_size=1, shuffle=True)

"""
构建模型
"""
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(13, 13),
            nn.Dropout(0.1),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(13, 13),
            nn.Dropout(0.1),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(13, 10),
            nn.Dropout(0.1),
            nn.ReLU()
        )

        self.fc4 = nn.Sequential(
            nn.Linear(10,5),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(5,1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        X = self.fc1(X)
        X = self.fc2(X)
        X = self.fc3(X)
        X = self.fc4(X)
        return self.sigmoid(X)

import torch.optim as optim

device = torch.device('cpu')

net = Net()
net.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

num_epochs = 20
def train():
    for epoch in range(num_epochs):
        train_loss_sum, train_total, train_accurate, start = 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            output = net(X)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_total += len(y)

        end = time.time()
        print( f"{epoch}: {train_loss_sum / train_total:.3f}, {end - start:.2f}")


train()

pred = net(test)
pred = pd.DataFrame(pred.tolist())
pred[0] = pred[0].apply(lambda x : 1 if x>=0.5 else 0)

submission['Survived'] = pred