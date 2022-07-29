import torch
import DIDLutils

X = torch.ones(6, 8)
K = torch.tensor([[1, -1]])
Y = DIDLutils.corr2d(X, K)
X[:, 2:6] = 0

conv2d = DIDLutils.Conv2D(kernel_size=(1,2))
step = 20
lr = 0.01
for i in range(step):
    Y_hat = conv2d(X)
    l = ((Y_hat - Y)**2).sum()
    l.backward()

    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.bias.data -= lr * conv2d.bias.grad

    conv2d.weight.grad.fill_(0)
    conv2d.bias.grad.fill_(0)

    if (i+1) % 5 == 0:
        print(f"Step {i+1}, loss {l.item()}")