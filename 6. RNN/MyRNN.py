import zipfile

import numpy as np
import torch
from torch import nn

import LoadLyrics

(lyrics_indices, char_to_idx, idx_to_char, vocab_size) = LoadLyrics.load_data_jay_lyrics()
"""
循环网络
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
自实现one_hot函数：将每一个字符所对应的索引映射为向量
"""


def one_hot(X, n_class, dtype=torch.float32):
    X = X.long()
    res = torch.zeros(X.shape[0], n_class, dtype=dtype, device=X.device)
    res.scatter_(1, X.view(-1, 1), 1)
    return res


# 测试以上函数
# x = torch.tensor([0, 2])
# print(one_hot(x, vocab_size))


def to_onehot(X, n_class):
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]


# 测试以上函数效果
X = torch.arange(10).view(2, 5)
# print(X)
# inputs = to_onehot(X, 10)
# for input in inputs:
#     print(input)

num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
print(f'将在{device}上训练')


def get_params():
    def _init_params(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)

    # 隐藏层参数
    W_xh = _init_params((num_inputs, num_hiddens))
    W_hh = _init_params((num_hiddens, num_hiddens))
    b_h = torch.nn.Parameter(torch.zeros(num_hiddens, device=device, requires_grad=True))
    # 输出层参数
    W_hq = _init_params((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, requires_grad=True))
    return nn.ParameterList([W_xh, W_hh, b_h, W_hq, b_q])


"""
定义模型
"""


def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device))


"""
下面的rnn函数定义了在一个时间步里如何计算隐藏状态和输出。
"""


def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H = state
    outputs = []
    for X in inputs:
        H = torch.tanh(X @ W_xh + H @ W_hh + b_h)
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return outputs, H


state = init_rnn_state(X.shape[0], num_hiddens, device)
inputs = to_onehot(X.to(device), vocab_size)
params = get_params()
outputs, state_new = rnn(inputs, state, params)


"""
定义预测函数
"""
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size, device, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens, device)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = to_onehot(torch.tensor([[output[-1]]], device=device), vocab_size)
        (Y, state) = rnn(X ,state, params)
        if t<len(prefix) - 1:
            output.append(char_to_idx[prefix[t+1]])
        else:
            output.append(int(Y[0].argmax(dim=1).item()))
    return ''.join([idx_to_char[i]for i in output])

print(predict_rnn('我是', 10, rnn, params, init_rnn_state, num_hiddens, vocab_size,
            device, idx_to_char, char_to_idx))