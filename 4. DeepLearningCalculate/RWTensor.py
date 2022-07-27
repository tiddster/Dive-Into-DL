import torch
from torch import nn

root_path = 'RWFile/'

"""
读写张量
"""
x = torch.ones(3)
torch.save(x, root_path+'x.pt')
x = torch.load(root_path+'x.pt')
print(x)

y = torch.zeros(4)
torch.save([x, y], root_path+'xy.pt')
xy = torch.load(root_path+'xy.pt')
print(xy)

torch.save({'x':x, 'y':y}, root_path+'xy_dict.pt')
xyDict = torch.load(root_path+'xy_dict.pt')
print(xyDict)

"""
读写模型
"""
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(784, 256)
        self.active = nn.ReLU()
        self.output = nn.Linear(256, 10)

    def forward(self, X):
        a = self.active(self.hidden(X))
        return self.output(a)

net = MLP()
print(net)
torch.save(net.state_dict(), root_path + "net.pt")

net = torch.load(root_path + "net.pt")
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer.state_dict()
print(optimizer)
