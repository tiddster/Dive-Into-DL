from random import randint

import torch
import torch.nn as nn
import torch.nn.functional as F

from Config import config
from DataProcess import id2word

output_file = "Dataset\\output.txt"

class GeneratorModule(nn.Module):
    def __init__(self, pretrain_model, config):
        super(GeneratorModule, self).__init__()
        self.config = config
        self.pretrain_model = pretrain_model  # 预训练的 LSTMCore
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y = self.pretrain_model(x)
        y_pred = self.pretrain_model.tag_space
        self.y_preb = self.softmax(y_pred)
        return self.y_preb

    def generate(self, start_tokens=None, batch_size=config.batch_size, seq_len=config.generate_seq_len):
        """
        这个函数没有开头的首字提示，在词汇表中随机寻找一个成为首字，进行随机生成序列以测试
        将y初始化为一个序列，和下面test函数保持一致
        :param start_tokens: (<=batch_size,   n),  n为每一个batch所拥有序列长度, 小于batch_size的那部分由模型自动生成
        :param batch_size:
        :param seq_len:
        :return:
        """
        samples = []
        for i in range(batch_size):
            if start_tokens is not None:
                if i < len(start_tokens):
                    y = start_tokens[i].tolist()
                else:
                    y = [randint(0, config.vocab_size)]
            else:
                y = [randint(0, config.vocab_size)]
            y_all_sample = y

            with torch.no_grad():
                # self.pretrain_model.hidden = self.pretrain_model.init_hidden()
                for i in range(seq_len - len(y)):
                    x = torch.tensor(y[-1]).int().view([-1, 1])
                    y_pred = self.pretrain_model(x)
                    y_pred = F.softmax(self.pretrain_model.tag_space, dim=1)
                    y_pred = y_pred.squeeze(dim=0)
                    y = y_pred.multinomial(num_samples=1)  # 按概率生成下一个字
                    y_all_sample.append(int(y.tolist()[0]))
            samples.append(y_all_sample)
        return torch.tensor(samples)


class LSTMCore(nn.Module):
    def __init__(self,config):
        super(LSTMCore, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.lstm = nn.LSTM(config.embedding_dim, config.hidden_dim)
        self.hidden2tag = nn.Linear(config.hidden_dim, config.vocab_size)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        """
        :param input:   [batch_size, seq_len]
        :return:
        """
        #  [batch_size, seq_len, emd_dim]
        emb_output = self.embedding(input)
        emb_output = emb_output.transpose(0, 1)

        # lstm_out: [batch_size, seq_len, hidden_dim]
        lstm_output, _ = self.lstm(emb_output)
        lstm_output = lstm_output.view(input.shape[0] * input.shape[1], -1)

        self.tag_space = self.hidden2tag(lstm_output)
        self.tag_scores = self.logSoftmax(self.tag_space)
        return self.tag_scores


def generate_sentences(model, generate_seq_num=100):
    samples = []
    for _ in range(generate_seq_num):
        sample = model.generate()
        samples.append(sample)
    return samples
