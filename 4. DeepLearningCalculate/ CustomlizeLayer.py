import torch
from torch import nn

X = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)

"""
不含模型参数的自定义层
"""


class Layer1(nn.Module):
    def __init__(self, **kwargs):
        super(Layer1, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()


layer = Layer1()
print(layer(X))

"""
含模型参数的自定义层
"""
# print(nn.Parameter(torch.randn(4, 4)))
X = torch.randn(4, 4)


class LayerWithParamslist(nn.Module):
    def __init__(self):
        super(LayerWithParamslist, self).__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.randn(4, 4)) for i in range(3)])
        self.params.append(nn.Parameter(torch.randn(4, 1)))

    def forward(self, x):
        for i in range(len(self.params)):
            x = torch.mm(x, self.params[i])
        return x


layer = LayerWithParamslist()
print(layer(X))
print(layer)

class LayerWithParamsDict(nn.Module):
    def __init__(self):
        super(LayerWithParamsDict, self).__init__()
        self.params = nn.ParameterDict({
            'linear1': nn.Parameter(torch.randn(4,4)),
            'linear2':nn.Parameter(torch.randn(4,4))
        })
        self.params.update({'linear3':nn.Parameter(torch.randn(4, 1))})

    def forward(self, X, choice='linear1'):
        return X @ self.params[choice]

layer = LayerWithParamsDict()
print(layer(X))
print(layer)