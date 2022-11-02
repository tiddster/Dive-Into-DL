import torch
import torch.nn as nn
import torch.nn.functional as F
from Config import config

class Generator(nn.torch):
    def __init__(self, config):
        self.Emb = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.Lstm = nn.LSTM(config.embedding_dim, config.hidden_dim)

        self.fc = nn.Linear(config.hidden_dim, 2)

    def forward(self, input):
        input = self.Emb(input)
        input = input.transpose(0, 1)

        output, (h, c) = self.Lstm(input)
        output = output.transpose(0, 1)

        output = self.fc(output)

        return F.softmax(output)