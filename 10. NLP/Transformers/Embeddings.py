import torch.nn as nn
import torch


class PositionEmbeddings(nn.Module):
    def __init__(self, max_len, model_dim):
        super(PositionEmbeddings, self).__init__()
        self.posEmb = torch.zeros(max_len, model_dim)
        self.posEmb.requires_grad = False

        pos = torch.arange(0, max_len).float().unsqueeze(1)

        _2i = torch.arange(0, model_dim, step=2).float()

        # a:b:c 从a到b，以c为步长的切片
        self.posEmb[:, 0::2] = torch.sin(pos / (10000 ** (_2i / model_dim)))
        self.posEmb[:, 1::2] = torch.cos(pos / (10000 ** (_2i / model_dim)))

    def forward(self, x):
        batch_size, seq_len, model_dim = x.shape

        # 复制batch_size份的位置信息
        # [batch_size, seq_len, model_dim]
        posInfo = self.posEmb[:seq_len, :]
        posInfo = posInfo.expand((batch_size, posInfo.shape[0], posInfo.shape[1]))

        return x + posInfo


# x = torch.rand((3, 10, 512))
# posEmb = PositionEmbeddings(500, 512)
# posEmb(x)
