import codecs
import time

import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, Dataset, DataLoader
from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
import pandas as pd

BERT_PATH = "F:\Dataset\Bert"
DATA_PATH = "F:\Dataset\\toutiao_cat_data\\toutiao_cat_data.txt"

"""
采用的是头条新闻数据
"""
# 用于获取text和对应的label
print("开始读取数据")
text_data, label_data, label_name = [], [], []
with open(DATA_PATH, "rt", encoding="utf-8") as f:
    for train_data in f.readlines():
        train_data = train_data.replace('\n', '').split('_!_')
        label_data.append(int(train_data[1]))
        label_name.append(train_data[2])
        text_data.append(train_data[3])


news_label = [int(x.split('_!_')[1])-100
                  for x in codecs.open(DATA_PATH, encoding='utf-8')]
news_text = [x.strip().split('_!_')[-1] if x.strip()[-3:] != '_!_' else x.strip().split('_!_')[-2]
                 for x in codecs.open(DATA_PATH, encoding='utf-8')]
x_train, x_test, train_label, test_label =  train_test_split(news_text[:50000], news_label[:50000], test_size=0.2, stratify=news_label[:50000])

index2label = set(label_name)
label2index = {label: index for index, label in enumerate(index2label)}


print(f"共有{len(train_label)}条训练数据，共有{len(test_label)}条测试数据")

print("正在进行数据编码")
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
train_encoding = tokenizer(x_train, truncation=True, padding=True, max_length=512)
test_encoding = tokenizer(x_test, truncation=True, padding=True, max_length=512)
print("数据编码完成")

class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        super(NewsDataset, self).__init__()
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[index]))
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_encoding, train_label)
test_dataset = NewsDataset(test_encoding, test_label)

"""
构建Bert模型
"""
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
model = BertForSequenceClassification.from_pretrained(BERT_PATH, num_labels=len(index2label))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

train_iter = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1)
test_iter = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=1)

optimizer = optim.SGD(model.parameters(), lr=2e-5)
num_epoch = 8
total_steps = len(label_data)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0, # Default value in run_glue.py
                                            num_training_steps=total_steps)

def train():
    model.train()
    train_loss_sum, train_total_num, train_correct_num, start, batch_num = 0.0, 0, 0, time.time(), 0

    for epoch in range(num_epoch):
        for batch in train_iter:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            train_loss_sum += loss.cpu().item()

            # 反向梯度信息
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 参数更新
            optimizer.step()
            scheduler.step()

            train_total_num += len(labels)
            train_correct_num += (outputs.argmax(dim=1) == labels).sum().cpu().item()

            batch_num += 1
            if batch_num % 200 == 0:
                print(f"epoch:{epoch}, batch_num:{batch_num}, loss:{loss}")
        end = time.time()
        print(f"epoch:{epoch}, batch_num:{batch_num}, epoch_loss:{train_loss_sum}, train_accuracy:{train_correct_num/train_total_num}, time:{end-start}")

if __name__ == "__main__":
    print("开始训练")
    torch.cuda.set_device(0)
    train()
