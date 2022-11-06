import torch
import torch.nn as nn
from torch.autograd import Variable
from Config import config

class PGLoss(nn.Module):
    """
    Pseudo-loss that gives corresponding policy gradients (on calling .backward())
    for adversial training of Generator
    """

    def __init__(self):
        super(PGLoss, self).__init__()

    def forward(self, pred, target, reward, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
            - pred: (batch_size, vocab_size),
            - target : (batch_size, seq_len ),
            - reward : (batch_size, seq_len)
        """
        loss = 0
        for i in range(config.batch_size):
            for j in range(config.generate_seq_len):
                loss += -pred[i][target[i][j]] * reward[i][j]  # log(P(y_t|Y_1:Y_{t-1})) * Q

        return loss / config.batch_size