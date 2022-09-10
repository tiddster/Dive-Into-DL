import os
import random

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

BERT_PATH = "F:\Dataset\Bert"
ROOT_PATH = "F:\Dataset\\NLPcc2013-2014Weibo_Emotion_Dataset\\Nlpcc2014\\Nlpcc2014Train_NoNone.tsv"

"""
获取数据集
"""

data = pd.read_csv(ROOT_PATH)

labels_in_text = list(data["标签"])  # 用文本表示的标签
text = list(data["文本"])

id2labels = set(labels_in_text)
labels2id = {label: index for index, label in enumerate(id2labels)}

labels_in_id = [labels2id[label] for label in labels_in_text]  # 用标签id表示标签

# 划分训练集和测试集
train_text, test_text, train_labels, test_labels = train_test_split(text, labels_in_id, test_size=0.2)

"""
利用bert构建词向量层
"""
from transformers import BertModel, BertTokenizer

def EmbeddingBlock(text):
    tokenizers = BertTokenizer.from_pretrained(BERT_PATH)
    return tokenizers(text, truncation=True, padding=True, max_length=64)

train_text_encoding, test_text_encoding = EmbeddingBlock(train_text), EmbeddingBlock(test_text)

print(EmbeddingBlock("今天天气真好"))

"""
自定义数据集
"""
from torch.utils.data import Dataset, DataLoader

class WeiboDataset(Dataset):
    def __init__(self, encoding, labels):
        super(WeiboDataset, self).__init__()
        self.encoding = encoding
        self.labels = labels

    def __getitem__(self, idx):
        token = self.encoding["input_ids"][idx]
        return token, self.labels[idx]

    def __len__(self):
        return len(self.labels)

train_dataset = WeiboDataset(train_text, train_labels)
test_dataset = WeiboDataset(test_text, test_labels)
train_iter = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_iter = DataLoader(test_dataset, batch_size=32)



