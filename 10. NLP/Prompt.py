import torch
import pandas as pd
import numpy as np

import DIDLutils

print(torch.cuda.is_available())
print(torch.cuda.get_device_name())
TRAIN_PATH = "F:/Dataset/Twitter2013/twitter-2013train-A.tsv"
TEST_PATH = "F:\Dataset\Twitter2013\\twitter-2013test-A.tsv"
BERT_PATH = "F:\Dataset\Bert-uncased"

train_data = pd.read_csv(TRAIN_PATH, sep='\t')
test_data = pd.read_csv(TEST_PATH, sep='\t')
train_labels_str, test_labels_str = train_data['labels'], test_data['labels']
train_text, test_text = train_data['text'], test_data['text']

idx2labels = set(train_labels_str)
labels2idx = {label: idx for idx, label in enumerate(idx2labels)}

train_labels, test_labels = [labels2idx[label] for label in train_labels_str], [labels2idx[label] for label in
                                                                                test_labels_str]

print(idx2labels)
print(f"数据读取完成，训练集个数{len(train_labels)}，测试集{len(test_labels)}")

"""
导入bert分词器
"""
from transformers import BertModel, BertConfig, BertTokenizer, get_cosine_schedule_with_warmup

tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

template = "It was Mask. "

pos_id = tokenizer.convert_tokens_to_ids('good')
neg_id = tokenizer.convert_tokens_to_ids('bad')
neu_id = tokenizer.convert_tokens_to_ids('neutral')

print(pos_id, neg_id, neu_id)


def get_embedding_inputs(text, labels):
    inputid_list = []
    labelid_list = []
    segid_list = []
    attid_list = []
    for i in range(len(text)):
        text_ = template + train_text[i]
        encode_dict = tokenizer.encode_plus(text_, max_length=60, padding='max_length', truncation=True)
        id = encode_dict["input_ids"]
        segment_id = encode_dict["token_type_ids"]
        attention_mask_id = encode_dict["attention_mask"]

        label_id, input_id = id[:], id[:]
        mask_pos = 3
        if labels[i] == 0:
            label_id[mask_pos] = pos_id
        elif labels[i] == 1:
            label_id[mask_pos] = neg_id
        else:
            label_id[mask_pos] = neu_id
        label_id[:mask_pos] = [-1] * len(label_id[:mask_pos])
        label_id[mask_pos + 1:] = [-1] * len(label_id[mask_pos + 1:])
        input_id[mask_pos] = tokenizer.mask_token_id

        inputid_list.append(input_id)
        labelid_list.append(label_id)
        segid_list.append(segment_id)
        attid_list.append(attention_mask_id)

    return inputid_list, labelid_list, segid_list, attid_list

print("开始创建数据集")
train_inputid_list, train_labelid_list, train_segid_list, train_attid_list = get_embedding_inputs(train_text,
                                                                                                  train_labels)
test_inputid_list, test_labelid_list, test_segid_list, test_attid_list = get_embedding_inputs(test_text,
                                                                                              test_labels)

print('1')
train_inputid_list, train_labelid_list, train_segid_list, train_attid_list = \
    torch.tensor(train_inputid_list, dtype=torch.long), torch.tensor(train_labelid_list, dtype=torch.long), \
    torch.tensor(train_segid_list, dtype=torch.long), torch.tensor(train_attid_list, dtype=torch.long)
print('2')
test_inputid_list, test_labelid_list, test_segid_list, test_attid_list = \
    torch.tensor(test_inputid_list, dtype=torch.long), torch.tensor(test_labelid_list, dtype=torch.long), \
    torch.tensor(test_segid_list, dtype=torch.long), torch.tensor(test_attid_list, dtype=torch.long)


"""
构建pytorch数据集
"""
from torch.utils.data import Dataset, DataLoader


class TwitterDataset(Dataset):
    def __init__(self, sen_tokens, att_mask, seg, label):
        super(TwitterDataset, self).__init__()
        self.sen_tokens = sen_tokens
        self.att_mask = att_mask
        self.seg = seg
        self.label = label

    def __len__(self):
        return self.sen_tokens.shape[0]

    def __getitem__(self, idx):
        return self.sen_tokens[idx], self.att_mask[idx], self.seg[idx], self.label[idx]


train_iter = DataLoader(TwitterDataset(train_inputid_list, train_attid_list, train_segid_list, train_labelid_list),
                           shuffle=True, batch_size=32)
test_iter = DataLoader(TwitterDataset(test_inputid_list, test_attid_list, test_segid_list, test_labelid_list),
                          shuffle=True, batch_size=32)

print("创建数据集完成")

"""
构建模型
"""
from torch import nn
from transformers import BertForMaskedLM, AdamW


class Bert_Model(nn.Module):
    def __init__(self, path, config):
        super(Bert_Model, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained(path, config=config)

    def forward(self, input_ids, att_mask_ids, seg_ids):
        outputs = self.bert(input_ids, att_mask_ids, seg_ids)
        logit = outputs[0]

        return logit


print("加载模型中")
config = BertConfig.from_pretrained(BERT_PATH)
net = Bert_Model(BERT_PATH, config)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = net.to(device)

import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=1e-6, weight_decay=1e-4, momentum=0.9)
loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
EPOCH_NUM = 20
schedule = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=len(train_iter),num_training_steps=EPOCH_NUM*len(train_iter))

"""
训练模型
"""
import time
def train():
    for epoch in range(EPOCH_NUM):
        train_loss_sum, train_total_num, train_correct_num, start, batch = 0.0, 0, 0, time.time(), 0
        net.train()
        for idx, (tokens, atts, segs, labels) in enumerate(train_iter):
            tokens, atts, segs, labels = tokens.to(device), atts.to(device), segs.to(device), labels.to(device)
            output = net(tokens, atts, segs)
            loss = loss_fn(output.view(-1, 30522), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            schedule.step()
            train_loss_sum += loss.item()
            batch += 1

            train_correct_num += (output.view(-1, 30522).argmax(dim=1) == labels.view(-1)).sum().item()
            train_total_num += labels.shape[0]

            if batch % 10 == 0:
                test_acc, test_loss = test_accuracy(net)
                DIDLutils.batch_print(epoch, batch, loss.item(), train_correct_num/train_total_num, test_acc, test_loss)

        test_acc, test_loss = test_accuracy(net)
        end = time.time()
        DIDLutils.epoch_print(epoch, train_loss_sum, train_correct_num/train_total_num, start, end, test_acc, test_loss)

def test_accuracy(net):
    eval_loss_sum = 0.0
    net.eval()
    correct_test = 0
    with torch.no_grad():
        for ids, att, tpe, y in test_iter:
            ids, att, tpe, y = ids.to(device), att.to(device), tpe.to(device), y.to(device)
            out_test = net(ids, att, tpe)
            loss_eval = loss_fn(out_test.view(-1, 30522), y.view(-1))
            eval_loss_sum += loss_eval.item()
            ttruelabel = y[:, 3]
            tout_train_mask = out_test[:, 3, :]
            predicted_test = torch.max(tout_train_mask.data, 1)[1]
            correct_test += (predicted_test == ttruelabel).sum()
            correct_test = np.float(correct_test)
    acc_test = float(correct_test / len(test_labels))
    return acc_test, eval_loss_sum


if __name__ == "__main__":
    train()
