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
        self.y_output = self.y_preb.multinomial(num_samples=1)
        return self.y_output

    def generate(self, start_token=None):
        if start_token:
            y = start_token
        else:
            y = randint(0, config.vocab_size)
        y_all_sample = [int(y)]
        with torch.no_grad():
            # self.pretrain_model.hidden = self.pretrain_model.init_hidden()
            for i in range(config.generate_seq_len - 1):
                x = torch.Tensor([y]).int().view([-1, 1])
                y_pred = self.pretrain_model(x)
                y_pred = F.softmax(self.pretrain_model.tag_space, dim=1)
                y_pred = y_pred.squeeze(dim=0)
                y = y_pred.multinomial(num_samples=1)  # 按概率生成下一个字
                y_all_sample.append(int(y.tolist()[0]))
        return y_all_sample


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


def generate_sentences(model):
    samples = []
    str_samples = []
    for _ in range(config.generate_seq_num):
        sample = model.generate()
        str_sample = [id2word[id] for id in sample]
        samples.append(sample)
        str_samples.append(str_sample)
    return samples, str_samples
    # with open(output_file, 'w') as fout:
    #     for sample in samples:
    #         string = ' '.join([str(s) for s in sample])
    #         fout.write('{}\n'.format(string))
