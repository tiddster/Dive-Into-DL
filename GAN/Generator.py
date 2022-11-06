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

    def forward(self, x):
        scores = self.pretrain_model(x)
        return scores

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
                    sample_tokens = start_tokens[i].tolist()
                else:
                    sample_tokens = [randint(0, config.vocab_size)]
            else:
                sample_tokens = [randint(0, config.vocab_size)]

            with torch.no_grad():
                # self.pretrain_model.hidden = self.pretrain_model.init_hidden()
                for j in range(seq_len - len(sample_tokens)):
                    temp_tokens = sample_tokens + [0] * (config.max_seqLen - len(sample_tokens))
                    x = torch.tensor(temp_tokens).int()
                    x = x.unsqueeze(0)
                    next_token_pred = self.pretrain_model(x)
                    next_token_pred = F.softmax(self.pretrain_model.output)
                    next_token_pred = next_token_pred.squeeze(dim=0)
                    next_token = next_token_pred.multinomial(num_samples=1)  # 按概率生成下一个字
                    sample_tokens.append(next_token)
            samples.append(sample_tokens)
        return torch.tensor(samples)


class LSTMCore(nn.Module):
    def __init__(self,config):
        super(LSTMCore, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.lstm = nn.LSTM(config.embedding_dim, config.hidden_dim)
        self.hidden2tag = nn.Linear(config.hidden_dim* config.max_seqLen, config.vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        """
        :param input:   [batch_size, seq_len]
        :return:
        """
        #  [batch_size, seq_len, emd_dim]
        emb_output = self.embedding(input)
        emb_output = emb_output.transpose(0, 1)

        # lstm_out: [seq_len, batch_size, hidden_dim]
        lstm_output, (_, _) = self.lstm(emb_output)
        # lstm_out: [batch_size, seq_len, hidden_dim]
        lstm_output = lstm_output.transpose(0, 1)
        lstm_output = lstm_output.reshape(input.shape[0], -1)

        self.output = self.hidden2tag(lstm_output)
        scores = self.softmax(self.output)
        return scores


def generate_sentences(model, generate_seq_num=100):
    samples = []
    for _ in range(generate_seq_num):
        sample = model.generate()
        samples.append(sample)
    return samples
