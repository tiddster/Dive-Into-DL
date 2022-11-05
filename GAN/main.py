import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from DataProcess import get_iter, pos_path, neg_path, id2word
from Generator import GeneratorModule, LSTMCore, generate_sentences
from Discriminater import DiscriminatorModule
from PGLoss import PGLoss
from Rollout import Rollout

from Config import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_generator_NLL(data_iter, criterion, optimizer):
    """
    单纯使用真实数据训练生成器
    """
    for epoch in range(config.epochs_nums):
        total_loss = 0.
        for data, labels in data_iter:
            data, labels = data.to(device), labels.to(device)
            labels = labels.contiguous().view(-1)
            output = generator(data)
            loss = criterion(output, labels)
            loss.requires_grad_(True)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss = total_loss / len(data_iter)
        print("G---------Epoch {}, train loss: {:.5f}".format(epoch, avg_loss))


def train_discriminator(data_iter, criterion, optimizer):
    for epoch in range(config.epochs_nums):
        total_loss = 0.
        for data, labels in data_iter:
            data, labels = data.to(device), labels.to(device)
            labels = labels.contiguous().view(-1)
            output = generator(data)
            loss = criterion(output, labels)
            loss.requires_grad_(True)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss = total_loss / len(data_iter)
        print("D---------Epoch {}, train loss: {:.5f}".format(epoch, avg_loss))

def train_generator_PG(gen, dis, rollout, pg_loss, optimizer):
    """
    1. 先生成batch_size个句子序列
    2. 将以上句子序列，从0到末尾以此放入鉴别器中，算出reward平均值
    3. 利用PG算法算出损失值
    """
    for epoch in range(config.epochs_nums):
        # construct the input to the genrator, add zeros before samples and delete the last column
        samples = generator.generate(batch_size=config.batch_size, seq_len=config.generate_seq_len)
        # zeros = torch.zeros(config.batch_size, 1, dtype=torch.int64)
        # zeros = zeros.to(device)

        inputs = samples
        targets = samples

        # calculate the reward
        rewards = torch.tensor(rollout.get_reward(samples, config.n_rollout, dis))
        rewards = rewards.to(device)

        # update generator
        output = gen(inputs)
        loss = pg_loss(output, targets, rewards)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("GD---------Epoch {}, train loss: {:.5f}".format(epoch, loss))


# 打印最终的完整句段和句子所对应的token
def generate_final_sentences(generator):
    token_samples = generate_sentences(generator, 200)
    texts = ""
    for tokens in token_samples:
        for token in tokens:
            texts += id2word[token]
        texts += '\n'

    with open(neg_path, 'w', encoding='utf8') as f:
        f.write(texts)



if __name__ == '__main__':
    pos_data_iter = get_iter()

    lstm = LSTMCore(config)
    generator = GeneratorModule(lstm, config)
    discriminator = DiscriminatorModule(config)
    rollout = Rollout(generator, config.update_rater)

    pg_criterion = PGLoss()
    nll_criterion = nn.NLLLoss()
    cel_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=generator.parameters(), lr=config.generator_lr)
    #
    for i in range(config.epochs_nums):
        train_generator_NLL(pos_data_iter, nll_criterion, optimizer)
        train_generator_PG(generator, discriminator, rollout, pg_criterion, optimizer)
        generate_final_sentences(generator)

        all_data_iter = get_iter(pos_path, neg_path)
        train_discriminator(all_data_iter, cel_criterion, optimizer)