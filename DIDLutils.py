from torch import nn
import torch
import matplotlib.pyplot as plt

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
训练模型
"""


def train(net, train_iter, test_iter, loss, num_epoch, batch_size, params=None, lr=None, optimizer=None):
    for epoch in range(num_epoch):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
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


"""
多层感知机板块-----------------------------------------------------------------
"""
"""
relu激活函数
"""


def relu(x):
    return torch.max(input=x, other=torch.tensor(0.0))


"""
K-折交叉验证集板块------------------------------------------------------------
"""


# 获取K-折交叉验证集， 将训练集分为K份，取其中第i份作为交叉验证集
# 其余作为真实训练集返回
def K_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        index = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[index, :], y[index]
        if j == i:
            X_cross, y_cross = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_cross, y_cross


def k_fold(net, train, k, X_train, y_train, num_epochs,
           learning_rate, weight_decay, batch_size):
    """
    :param net:  网络模型
    :param train:   训练函数
    :param k:   K折
    :param X_train:  训练集的特征
    :param y_train:   训练集的标签
    :return:
    """
    train_loss_sum, cross_loss_sum = 0, 0
    for i in range(k):
        data = K_fold_data(k, i, X_train, y_train)
        train_loss, cross_loss = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_loss_sum += train_loss[-1]
        cross_loss_sum += cross_loss[-1]
        plt.plot(range(1, num_epochs+1), train_loss)
        plt.plot(range(1, num_epochs + 1), cross_loss)
        plt.show()
        print(f'fold {i}, train rmse {train_loss[-1]}, cross rmse {cross_loss[-1]}')
    return train_loss_sum / k, cross_loss_sum / k
