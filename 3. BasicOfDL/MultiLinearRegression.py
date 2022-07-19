import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import DIDLutils as dl

n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05
features = torch.randn((n_train + n_test, num_inputs))
labels = features @ true_w + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]

w = torch.randn((num_inputs, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
params = [w, b]

"""
定义L2惩罚项以避免过拟合、欠拟合
"""


def l2_penalty(w):
    return (w ** 2).sum() / 2


batch_size, num_epochs, lr = 1, 100, 0.03
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
)
loss = nn.MSELoss()

dataset = Data.TensorDataset(train_features, train_labels)
train_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

def fit_and_plot_pytorch(weight_decay):
    net = nn.Linear(num_inputs, 1)
    nn.init.normal_(net.weight, mean=0, std=1)
    nn.init.normal_(net.bias, mean=0, std=1)
    optimizer_w = torch.optim.SGD(params=[net.weight], lr=lr, weight_decay=weight_decay)  # 对权重参数衰减
    optimizer_b = torch.optim.SGD(params=[net.bias], lr=lr)  # 不对偏差参数衰减

    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y).mean()
            optimizer_w.zero_grad()
            optimizer_b.zero_grad()

            l.backward()

            # 对两个optimizer实例分别调用step函数，从而分别更新权重和偏差
            optimizer_w.step()
            optimizer_b.step()
        print('L2 norm of w:', net.weight.data.norm().item())

fit_and_plot_pytorch(0.01)