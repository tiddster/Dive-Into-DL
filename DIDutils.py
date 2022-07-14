from torch import nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.get_device_name())

""""
输入层的形状转换
"""
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


"""
softmax回归板块------------------------------------------------------------
"""
"""
softmax模型网络：
softmax横向相加进行归一化处理
再利用网络模型构建全连接层
"""


def softmax(X):
    X_exp = X.exp()
    X_reg = X_exp / X_exp.sum(dim=1, keepdims=True)
    return X_reg


def softmaxNet(w, x, b, num_inputs):
    """
    :param w:   权重的维度是 x特征数 * output个数
    :param x:   维度是 input个数 * x特征数
    :param b:  维度是 1 * output个数
    :param num_inputs:
    :return:
    """
    O = x.view((-1, num_inputs)) @ w + b
    return softmax(O)


"""
交叉熵损失函数
"""
def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))


"""
准确率函数
"""
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

"""
训练softmax模型
"""
def trainSoftmax(net, train_iter, test_iter, loss, num_epoch, batch_size, params=None, lr = None, optimizer = None):
    for epoch in range(num_epoch):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and param[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()

            if optimizer is None:
                optimizer = torch.optim.SGD(net.parameters(), lr=lr)
            else:
                optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
