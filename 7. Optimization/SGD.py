import time

import numpy as np
import torch.utils.data
from matplotlib import pyplot as plt
from torch import nn, optim

import  LoadAirfoilNoiseDataset


def sgd(params, states, hyperparams):
    for p in params:
        p.data -= hyperparams['lr'] * p.grad.data

def train_by_sgd(optimizer_fn, optimizer_hyperparams, features, labels,
                    batch_size=10, num_epochs=2):
    net, loss = nn.Sequential(nn.Linear(features.shape[-1], 1)), nn.MSELoss()
    optimizer = optimizer_fn(net.parameters(), **optimizer_hyperparams)

    def eval_loss():
        return loss(net(features).view(-1), labels).item() / 2

    ls = [eval_loss()]
    data_iter = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(features, labels), batch_size, shuffle=True)

    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            l = loss(net(X).view(-1), y)/2

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())
        print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
        plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    plt.show()

features, labels = LoadAirfoilNoiseDataset.feature, LoadAirfoilNoiseDataset.labels
train_by_sgd(optim.SGD, {"lr": 0.05}, features, labels, num_epochs=10)