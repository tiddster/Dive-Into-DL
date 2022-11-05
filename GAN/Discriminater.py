import torch
import torch.nn as nn
import torch.nn.functional as F

from Config import config

class DiscriminatorModule(nn.Module):
    """A CNN for text classification
    architecture: Embedding >> Convolution >> Max-pooling >> Softmax
    """

    def __init__(self, config):
        super(DiscriminatorModule, self).__init__()
        self.Emb = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, n, (f, config.embedding_dim)) for (n, f) in zip(config.num_filters, config.filter_sizes)]
        )
        self.highway = nn.Linear(sum(config.num_filters), sum(config.num_filters))
        self.fc = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(sum(config.num_filters), 2)
        )

    def forward(self, input):
        emb_output = self.Emb(input).unsqueeze(1)   # [batch_size, 1 , seq_len , emb_dim]
        conv_output = [F.relu(conv(emb_output)).squeeze(3) for conv in self.convs] # [batch_size, num_filter, len]

        pool_output = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in conv_output]  # [batch_size, num_filter]

        pred_output = torch.cat(pool_output, 1)

        highway_output = self.highway(pred_output)
        pred = torch.sigmoid(highway_output) * F.relu(highway_output) + (1. - torch.sigmoid(highway_output)) * pred_output
        pred = self.fc(pred)

        return F.softmax(pred)
