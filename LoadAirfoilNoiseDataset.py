import numpy as np
import torch


def get_data():
    data = np.genfromtxt("F:\MINSTDataset\\NASA\\airfoil_self_noise.dat", delimiter='\t')
    data = data - data.mean(axis=0)/data.std(axis=0)
    return torch.tensor(data[:1500, :-1], dtype=torch.float32), torch.tensor(data[:1500, :-1])

feature, labels = get_data()
print(feature.shape)