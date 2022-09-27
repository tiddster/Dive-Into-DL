import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')

"""
构建数据集
"""

# 3 words sentences (=sequence_length is 3)
sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.

vocab = list(set(" ".join(sentences).split()))
word2idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(word2idx)


def make_data(sentences, labels):
    inputs = []
    for sen in sentences:
        inputs.append(np.asarray([word2idx[word] for word in sen.split()]))

    return torch.LongTensor(inputs), torch.LongTensor(labels)


BATCH_SIZE = 1

inputs, labels = make_data(sentences, labels)
dataset = TensorDataset(inputs, labels)
train_iter = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


"""
构建模型
"""
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

embedding_dim = 2
n_hidden = 5  # number of hidden units in one cell
num_classes = 2

class BiLSTM_Attention(nn.Module):
    def __init__(self):
        super(BiLSTM_Attention, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.BiLSTM = nn.LSTM(embedding_dim, n_hidden, bidirectional=True)
        self.fc = nn.Linear(n_hidden * 2, num_classes)

    def Attenion(self, lstm_output, final_state):
        # lstm_output : [batch_size, seq_len, num_hidden * num_directions(=2)], F matrix
        # final_state : [num_directions(=2), batch_size, num_hidden]
        batch_size = len(lstm_output)
        # hidden=[batch_size, num_hidden*num_directions(=2), 1]
        hidden = final_state.view((batch_size, -1, 1))

        # torch.bmm为多维矩阵的乘法：a=[b, h, w], c=[b,w,m]  bmm(a,b)=[b,h,m], 也就是对每一个batch都做矩阵乘法
        # squeeze(2), 判断第三维上维度是否为1，若为1则去掉
        # attn_weights:
        # = [batch_size, seq_len, num_hidden * num_directions(=2)] @  [batch_size, num_hidden*num_directions(=2), 1]
        # = [batch_size, seq_len, 1]
        attn_weights = lstm_output @ hidden

        soft_attn_weights = F.softmax(attn_weights, 1)

        # context
        # = [batch_size, num_hidden * num_directions(=2), seq_len] @  [batch_size, seq_len, 1]
        # = [batch_size, num_hidden * num_directions]
        context = (lstm_output.transpose(1, 2) @ soft_attn_weights).squeeze(2)

        return context, soft_attn_weights

    def forward(self, X):
        """
        :param X:[batch_size, seq_len]
        :return:
        """
        # inputs: [batch_size, seq_len, embedding_dim]
        inputs = self.embeddings(X)
        # inputs: [seq_len, batch_size, embedding_dim]
        inputs = inputs.transpose(0,1)

        output, (final_hidden_state, final_cell_state) = self.BiLSTM(inputs)
        # output : [batch_size, seq_len, n_hidden * num_directions(=2)]
        # final_hidden_state : [num_directions, batch_size, num_hidden]
        output = output.transpose(0, 1)
        attn_output, attention = self.Attenion(output, final_hidden_state)

        # attn_output : [batch_size, num_classes], attention : [batch_size, seq_len, 1]
        return self.fc(attn_output),attention


net = BiLSTM_Attention().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

def train():
    for epoch in range(100):
        for x, y in train_iter:
            x, y = x.to(device), y.to(device)
            pred, _ = net(x)
            loss = criterion(pred, y)
            if (epoch + 1) % 1000 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

train()

test_text_list = ['i hate baseball', 'i love that']
tests = [[word2idx[word] for word in test_text.split()] for test_text in test_text_list]
test_batch = torch.LongTensor(tests).to(device)

# Predict
predict, _ = net(test_batch)
predict = predict.argmax(dim=1)
print(predict)



