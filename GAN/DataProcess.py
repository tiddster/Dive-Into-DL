from torch.utils.data import Dataset, DataLoader
import torch
import os
import json
import re

pos_path = "P:\Dataset\ChinesePoetry\poetry"
json_path = "P:\Dataset\ChinesePoetry\poetry"
neg_path = "Dataset\\output.txt"

max_seqLen = 5


def get_json_data(path=json_path):
    fileNames = get_json_file_name(path)
    textList = []
    for name in fileNames[:2000]:
        data = get_json_single_data(name)
        text = ''
        for d in data:
            if d == '，' or d == '。' or d == '.' or d == '？':
                textList.append(text)
                text = ''
            else:
                text += d
    return textList


def get_json_single_data(json_file):
    fullPath = json_path + f'\\{json_file}'
    f = open(fullPath, 'r', encoding='utf8')
    json_data = json.load(f)
    return json_data['content'].replace('<br>','').replace(' ','').replace('《','').replace('》','')\
        .replace('；','').replace('！','').replace('“','').replace('”','').replace('：','').replace('\u3000','')\
        .replace('<p>', '').replace('</p>','').replace('、','').replace('<divid="shicineirong"class="shicineirong">', '')\
        .replace('’','').replace('‘','').replace('_','')


def get_json_file_name(path=json_path):
    fileName = os.listdir(path)
    return fileName


def get_data(path):
    with open(path, 'r', encoding='utf8') as f:
        textList = []
        for text in f.readlines():
            text = text.replace('\n', '').replace('\ufeff', '').replace('1', '').replace(' ', '').replace('-', '')
            textList.append(text)
        return textList


def get_vocab(textList):
    id2word = ['<pad>']
    for text in textList:
        texts = list(set(text))
        id2word += texts
    id2word = list(set(id2word))
    word2id = {word: i for i, word in enumerate(id2word)}
    return id2word, word2id


def get_token(textList, word2id):
    tokenList = []
    for text in textList:
        tokens = []
        text = re.sub('[a-zA-Z]', '', text)
        for t in text:
            if t != '.' and '\\u' not in t and t != '_':
                tokens.append(word2id[t])
        tokenList.append(tokens)
    return tokenList


id2word, word2id = get_vocab(get_json_data(json_path))


def get_iter(pos_path=json_path, neg_path=None):
    if neg_path:
        neg_text_list = get_data(neg_path)
        neg_labels = [0 for _ in range(len(neg_text_list))]
    else:
        neg_text_list = []
        neg_labels = []

    pos_text_list = get_json_data(pos_path)
    pos_labels = [1 for _ in range(len(pos_text_list))]

    textList = pos_text_list + neg_text_list
    labels = pos_labels + neg_labels

    tokenList = get_token(textList, word2id)
    # print(len(tokenList))

    for i in range(len(tokenList)):
        if len(tokenList[i]) > max_seqLen:
            tokenList[i] = tokenList[i][:max_seqLen]
        else:
            tokenList[i] = tokenList[i] + [0] * (max_seqLen - len(tokenList[i]))

    # finalList = []
    # for tokens in tokenList:
    #     for t in tokens:
    #         finalList.append(t)

    dataset = GANDataset(tokenList, labels)
    iter = DataLoader(dataset, batch_size=32, shuffle=True)
    return iter


class GANDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x).long()
        self.y = torch.tensor(y).long()

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.y.shape[0]
