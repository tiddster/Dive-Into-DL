import torch
import torch.nn as nn
from torch.autograd import Variable

class PGLoss(nn.Module):
    """
    Pseudo-loss that gives corresponding policy gradients (on calling .backward())
    for adversial training of Generator
    """

    def __init__(self):
        super(PGLoss, self).__init__()

    def forward(self, pred, target, reward):
        """
            - pred: (batch_size, seq_len),
            - target : (batch_size, ),
            - reward : (batch_size, )
        """
        N = target.size(0)
        C = pred.size(1)
        one_hot = torch.zeros((N, C))
        if pred.is_cuda:
            one_hot = one_hot.cuda()
        one_hot.scatter_(1, target.data.view((-1, 1)), 1)
        one_hot = one_hot.type(torch.ByteTensor)
        one_hot = Variable(one_hot)
        if pred.is_cuda:
            one_hot = one_hot.cuda()
        loss = torch.masked_select(pred, one_hot)
        loss = loss * reward
        loss = -torch.sum(loss)
        return loss