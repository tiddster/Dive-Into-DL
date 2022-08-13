import time
import torch
from torch import nn
import DIDLutils
import LoadLyrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lyrics_indices, char_to_idx, idx_to_char, vocab_size = LoadLyrics.load_data_jay_lyrics()

num_hiddens = 256
rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)

num_steps = 35
batch_size = 2
state = None
X = torch.rand(num_steps, batch_size, vocab_size)
Y, state_new = rnn_layer(X, state)

model = DIDLutils.RNNModel(rnn_layer, vocab_size).to(device)
DIDLutils.predict_rnn_pytorch('分开', 10, model, vocab_size, device, idx_to_char, char_to_idx)

num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e-3, 1e-2 # 注意这里的学习率设置
pred_period, pred_len, prefixes = 50, 50, ['我', '双节']
DIDLutils.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                            lyrics_indices, idx_to_char, char_to_idx,
                            num_epochs, num_steps, lr, clipping_theta,
                            batch_size, pred_period, pred_len, prefixes)