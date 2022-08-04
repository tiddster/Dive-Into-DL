import time

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

"""
卷积神经网络板块------------------------------------------------------------
"""
# 卷积操作
def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0]-h+1, X.shape[1]-w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y

# 二维卷积层
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, X):
        return corr2d(X, self.weight) + self.bias

# 多通道输入
def corr2d_multi_in(X, K):
    res = corr2d(X[0, :, :], K[0, :, :])
    for i in range(1, X.shape[0]):
        res +=  corr2d(X[i, :, :], K[i, :, :])
    return res

# 多通道输出
def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K])


# 1*1多通道输出
def corr2d_multi_in_out_1_1(X, K):
    c_in, h, w = X.shape
    c_out = K.shape[0]
    X = X.view(c_in, h*w)
    K = K.view(c_out, c_in)
    Y = torch.mm(K, X)
    return Y.view(c_out, h, w)

# 池化操作
PoolMax = 'max'
PoolAvg = 'avg'
def pool2d(X, pool_size, mode=PoolMax):
    X = X.float()
    pool_h, pool_w = pool_size
    Y = torch.zeros(X.shape[0] - pool_h+1, X.shape[1] - pool_w+1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == PoolMax:
                Y[i, j] = X[i:i+pool_h, j:j+pool_w].max()
            elif mode == PoolAvg:
                Y[i, j] = X[i:i+pool_h, j:j+pool_w].mean()

"""
卷积神经网络LeNet板块
"""
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()  # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()  # 改回训练模式
            else:  # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if ('is_training' in net.__code__.co_varnames):  # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

def train_CNNet(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))