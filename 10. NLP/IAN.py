import time
from collections import Counter

import numpy as np
import spacy


class Config():
    def __init__(self, max_aspect_len, max_context_len, vocab_size, embedding=None):
        self.max_aspect_len = max_aspect_len
        self.max_context_len = max_context_len
        self.vocab_size = vocab_size
        self.embedding = embedding
        self.embedding_size = 300
        self.hidden_size = 300
        self.lr = 0.03
        self.n_class = 3
        self.dropout = 0.01
        self.clip = 3


"""
读取数据
"""

nlp = spacy.load("en_core_web_sm")
root_path = "F:\IAN-pytorch-master\IAN-pytorch-master\data\laptop\\"


def read_txt(file):
    words, labels = [], []
    max_context_len, max_aspect_len = 0, 0
    lines = open(file, 'r').readlines()
    for i in range(0, len(lines), 3):
        # 处理第一行文本
        sp_words = nlp(lines[i].strip())  # .strip()  去除指定字符
        words.extend([sp.text.lower() for sp in sp_words])
        if len(sp_words) - 1 > max_context_len:
            max_context_len = len(sp_words) - 1

        # 处理第二行aspect
        sp_aspect = nlp(lines[i + 1].strip())
        words.extend([sp.text.lower() for sp in sp_aspect])
        if len(sp_aspect) > max_aspect_len:
            max_aspect_len = len(sp_aspect)
    return words, max_context_len, max_aspect_len


# 获取数据 生成word2id字典
def get_data_info(root_path):
    train_file = root_path + 'train.txt'
    test_file = root_path + 'test.txt'
    save_file = root_path + 'save_vocab.txt'

    word2id, max_aspect_len, max_context_len = {}, 0, 0
    word2id['<pad>'] = 0

    train_words, train_max_context, train_max_aspect = read_txt(train_file)
    test_words,  test_max_context, test_max_aspect = read_txt(test_file)
    max_aspect_len = max(train_max_aspect, test_max_aspect)
    max_context_len = max(train_max_context, test_max_context)

    words = train_words + test_words

    word_count = Counter(words).most_common()
    for word, _ in word_count:
        if word not in word2id and ' ' not in word and '\n' not in word and 'aspect_term' not in word:
            word2id[word] = len(word2id)

    with open(save_file, 'w') as f:
        f.write('length %s %s\n' % (max_aspect_len, max_context_len))
        for key, value in word2id.items():
            f.write('%s %s\n' % (key, value))

    print('There are %s words in the dataset, the max length of aspect is %s, and the max length of context is %s' % (
        len(word2id), max_aspect_len, max_context_len))
    return word2id, max_aspect_len, max_context_len


# 读取数据并转换为词向量
def read_data_info(word2id, max_aspect_len, max_context_len, root_path, file_name):
    file = root_path + file_name

    aspects, contexts, labels, aspect_lens, context_lens = list(), list(), list(), list(), list()

    lines = open(file, 'r').readlines()
    for i in range(0, len(lines), 3):
        polarity = lines[i + 2].split()[0]
        if polarity == 'conflict':
            continue

        context_sp_words = nlp(lines[i].strip())
        context = []
        for sp in context_sp_words:
            if sp.text.lower() in word2id:
                context.append(word2id[sp.text.lower()])

        aspect_sp_words = nlp(lines[i + 1].strip())
        aspect = []
        for sp in aspect_sp_words:
            if sp.text.lower() in word2id:
                aspect.append(word2id[sp.text.lower()])

        aspects.append(aspect + [0] * (max_aspect_len - len(aspect)))
        contexts.append(context + [0] * (max_context_len - len(context)))
        if polarity == 'negative':
            labels.append(0)
        elif polarity == 'neutral':
            labels.append(1)
        elif polarity == 'positive':
            labels.append(2)
        aspect_lens.append(len(aspect_sp_words))
        context_lens.append(len(context_sp_words) - 1)
    print("Read %s examples from %s" % (len(aspects), file))
    aspects = np.asarray(aspects)
    contexts = np.asarray(contexts)
    labels = np.asarray(labels)
    aspect_lens = np.asarray(aspect_lens)
    context_lens = np.asarray(context_lens)
    data = {"aspects": aspects, "contexts": contexts, "labels": labels,
            "aspect_lens": aspect_lens, "context_lens": context_lens}
    return data

# 读取embedding
def load_word_embeddings(embedding_dim, word2id):
    file = "F:\Dataset\glove.840B.300d\glove.840B.300d\\glove.840B.300d.txt"

    word2vec = np.random.uniform(-0.01, 0.01, [len(word2id), embedding_dim])
    oov = len(word2id)
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.split(' ')
            if content[0] in word2id:
                word2vec[word2id[content[0]]] = np.array(list(map(float, content[1:])))
                oov = oov - 1
    word2vec[word2id['<pad>'], :] = 0
    print('There are %s words in vocabulary and %s words out of vocabulary' % (len(word2id) - oov, oov))
    return word2vec


from torch.utils.data import Dataset, DataLoader


class IanDataset(Dataset):
    def __init__(self, data):
        self.aspects = torch.from_numpy(data['aspects']).long()
        self.contexts = torch.from_numpy(data['contexts']).long()
        self.labels = torch.from_numpy(data['labels']).long()
        self.aspect_lens = torch.from_numpy(data['aspect_lens']).long()
        self.context_lens = torch.from_numpy(data['context_lens']).long()
        self.len = self.labels.shape[0]
        aspect_max_len = self.aspects.size(1)
        context_max_len = self.contexts.size(1)
        self.aspect_mask = torch.zeros(aspect_max_len, aspect_max_len)
        self.context_mask = torch.zeros(context_max_len, context_max_len)
        for i in range(aspect_max_len):
            self.aspect_mask[i, 0:i + 1] = 1
        for i in range(context_max_len):
            self.context_mask[i, 0:i + 1] = 1

    def __getitem__(self, index):
        return self.aspects[index], self.contexts[index], self.labels[index], \
               self.aspect_mask[self.aspect_lens[index] - 1], self.context_mask[self.context_lens[index] - 1]

    def __len__(self):
        return self.len

import torch
import torch.nn as nn

"""
构建IAN模型
"""


class Attention(nn.Module):
    def __init__(self, q_size, k_size):
        super(Attention, self).__init__()
        self.weights = nn.Parameter(torch.rand(q_size, k_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, q, k, m):
        # query: (batch_size, query_size)
        # key: (batch_size, time_step, key_size)
        # value: (batch_size, time_step, value_size)
        # mask: (batch_size, time_step)
        batch_size = k.size(0)
        time_step = k.size(1)
        weights = self.weights.repeat(batch_size, 1, 1)  # (batch_size, key_size, query_size)
        query = q.unsqueeze(-1)  # (batch_size, query_size, 1)
        mids = weights @ query  # (batch_size, key_size, 1)
        mids = mids.repeat(time_step, 1, 1, 1).transpose(0, 1)  # (batch_size, time_step, key_size, 1)
        key = k.unsqueeze(-2)  # (batch_size, time_step, 1, key_size)
        scores = torch.tanh((key @ mids).squeeze() + self.bias)  # (batch_size, time_step, 1, 1)
        scores = scores.squeeze()  # (batch_size, time_step)
        scores = scores - scores.max(dim=1, keepdim=True)[0]
        scores = torch.exp(scores) * m
        attn_weights = scores / scores.sum(dim=1, keepdim=True)
        return attn_weights


class IAN(nn.Module):
    def __init__(self, config):
        super(IAN, self).__init__()
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self.n_class = config.n_class
        self.max_aspect_len = config.max_aspect_len
        self.max_context_len = config.max_context_len

        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size)
        self.aspect_lstm = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, batch_first=True)
        self.context_lstm = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, batch_first=True)
        self.aspect_attn = Attention(self.hidden_size, self.hidden_size)
        self.context_attn = Attention(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(self.hidden_size * 2, self.n_class)
        self.embedding.weight.data.copy_(torch.from_numpy(config.embedding))

    def forward(self, aspect, context, aspect_mask, context_mask):
        aspect = self.embedding(aspect)
        aspect = self.dropout(aspect)
        aspect_output, _ = self.aspect_lstm(aspect)
        aspect_output = aspect_output * aspect_mask.unsqueeze(-1)
        aspect_avg = aspect_output.sum(dim=1, keepdim=False) / aspect_mask.sum(dim=1, keepdim=True)

        context = self.embedding(context)
        context = self.dropout(context)
        context_output, _ = self.context_lstm(context)
        context_output = context_output * context_mask.unsqueeze(-1)
        context_avg = context_output.sum(dim=1, keepdim=False) / context_mask.sum(dim=1, keepdim=True)

        aspect_attn = self.aspect_attn(context_avg, aspect_output, aspect_mask).unsqueeze(1)
        aspect_features = (aspect_attn @ aspect_output).squeeze()

        context_attn = self.context_attn(aspect_avg, context_output, context_mask).unsqueeze(1)
        context_features = (context_attn @ context_output).squeeze()

        features = torch.cat([aspect_features, context_features], dim=1)
        features = self.dropout(features)
        output = self.fc(features)
        output = torch.tanh(output)
        return output


import torch.optim as optim


def train():
    for epoch in range(50):
        start_time = time.time()
        train_total_cases = 0
        train_correct_cases = 0
        for data in train_loader:
            aspects, contexts, labels, aspect_masks, context_masks = data
            aspects, contexts, labels = aspects.cuda(), contexts.cuda(), labels.cuda()
            aspect_masks, context_masks = aspect_masks.cuda(), context_masks.cuda()
            outputs = net(aspects, contexts, aspect_masks, context_masks)
            _, predicts = outputs.max(dim=1)

            train_total_cases += labels.shape[0]
            train_correct_cases += (predicts == labels).sum().item()

            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), config.clip)
            optimizer.step()

        train_accuracy = train_correct_cases / train_total_cases

        test_total_cases = 0
        test_correct_cases = 0
        for data in test_loader:
            aspects, contexts, labels, aspect_masks, context_masks = data
            aspects, contexts, labels = aspects.cuda(), contexts.cuda(), labels.cuda()
            aspect_masks, context_masks = aspect_masks.cuda(), context_masks.cuda()
            outputs = net(aspects, contexts, aspect_masks, context_masks)
            _, predicts = outputs.max(dim=1)
            test_total_cases += labels.shape[0]
            test_correct_cases += (predicts == labels).sum().item()

        test_accuracy = test_correct_cases / test_total_cases
        print('[epoch %03d] train accuracy: %.4f test accuracy: %.4f' % (epoch, train_accuracy, test_accuracy))
        end_time = time.time()
        print('Time Costing: %s' % (end_time - start_time))


if __name__ == "__main__":
    print("开始读取数据")
    word2id, max_aspect_len, max_context_len = get_data_info(root_path)
    config = Config(max_aspect_len, max_context_len, len(word2id))
    config.embedding = load_word_embeddings(config.embedding_size, word2id)
    train_data = read_data_info(word2id, max_aspect_len, max_context_len, root_path, 'train.txt')
    test_data = read_data_info(word2id, max_aspect_len, max_context_len, root_path, 'test.txt')
    train_dataset = IanDataset(train_data)
    test_dataset = IanDataset(test_data)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False, num_workers=2)
    print("读取数据完成")
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    net = IAN(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=config.lr)
    print("开始训练")
    train()
