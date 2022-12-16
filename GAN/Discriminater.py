import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

class DiscriminatorModule(nn.Module):

    def __init__(self, config, dropout=0.2):
        super(DiscriminatorModule, self).__init__()
        self.hidden_dim = config.hidden_dim
        self.embedding_dim = config.embedding_dim
        self.max_seq_len = config.max_seqLen
        self.config = config

        self.embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.gru = nn.GRU(config.embedding_dim, config.hidden_dim, num_layers=2, bidirectional=True, dropout=dropout)
        self.gru2hidden = nn.Linear(2*2*config.hidden_dim, config.hidden_dim)
        self.dropout_linear = nn.Dropout(p=dropout)
        self.hidden2out = nn.Linear(config.hidden_dim, 2)

    def init_hidden(self, batch_size):
        h = autograd.Variable(torch.zeros(2*2*1, batch_size, self.hidden_dim)).to(self.config.device)
        return h

    def forward(self, input):
        # input dim                                                # batch_size x seq_len
        hidden = self.init_hidden(input.size()[0])

        # [batch_size, seq_len, embedding_dim]
        emb = self.embeddings(input)
        # [seq_len, batch_size, embedding_dim]
        emb = emb.permute(1, 0, 2)
        # [4, batch_size, hidden_dim]
        _, hidden = self.gru(emb, hidden)

        # [batch_size, 4, hidden_dim]
        hidden = hidden.permute(1, 0, 2).contiguous()
        # [batch_size, hidden_dim]
        out = self.gru2hidden(hidden.view(-1, 4*self.hidden_dim))

        out = torch.tanh(out)
        out = self.dropout_linear(out)

        # [batch_size, 2]
        out = self.hidden2out(out)
        out = F.softmax(out)

        return out