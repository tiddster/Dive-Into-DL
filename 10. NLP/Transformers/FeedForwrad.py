import torch.nn as nn


class FeedForwardNet(nn.Module):
    def __init__(self, model_dim):
        super(FeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim)
        )
        self.layerNorm = nn.LayerNorm(model_dim)

    def forward(self, X):
        res = X
        X = self.fc(X)

        return self.layerNorm(res + X)
