import collections
import os
import random

import torch
from torch import nn

"""
数据集采用 斯坦福大学IMDB影评
"""

ROOT_PATH = "F:\Dataset\\aclImdb_v1\\aclImdb"

def read_imdb(folder_name):
    data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join(ROOT_PATH, folder_name, label)
        for file in folder_name:
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '').lower()
                data.append([review, 1 if label == 'pos' else 0])
    random.shuffle(data)
    return data

train_data, test_data = read_imdb('train'), read_imdb('test')
