import torch
import torch.nn as nn
from  Config import  config

class LSTM(nn.Module):
    def __init__(self):
        self.Emb = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.Lstm = nn.LSTM(config.embedding_dim, config.hidden_dim)
        self.fc = nn.Linear(config.hidden_dim, 2)

    def forward(self, x):
        x = self.Emb(x)

