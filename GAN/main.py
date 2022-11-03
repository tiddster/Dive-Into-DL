import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from DataProcess import get_iter
from Generator import GeneratorModule, LSTMCore, generate_sentences
from Discriminater import DiscriminatorModule
from PGLoss import PGLoss

from Config import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_generator_PG(generator, criminator, rollout, pg_loss, optimizer):
    """
    Train generator with the guidance of policy gradient
    """
    for epoch in range(config.epochs_nums):
        # construct the input to the genrator, add zeros before samples and delete the last column
        samples = generator.sample()
        zeros = torch.zeros(config.batch_size, 1, dtype=torch.int64).to(device)
        inputs = torch.cat([zeros, samples.data], dim=1)[:, :-1].contiguous()
        targets = samples.data.contiguous().view((-1,))

        # calculate the reward
        rewards = torch.tensor(rollout.get_reward(samples, config.n_rollout, criminator))
        rewards = rewards.to(device)

        # update generator
        output = generator(inputs)
        loss = pg_loss(output, targets, rewards)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_generator_MLE(data_iter, criterion, optimizer):
    """
    Train generator with MLE
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
        print("Epoch {}, train loss: {:.5f}".format(epoch, avg_loss))


# 打印最终的完整句段和句子所对应的token
def generate_final_sentences(generator):
    token_samples, str_samples = generate_sentences(generator)
    for t in token_samples:
        print(t)
    for strList in str_samples:
        res = ""
        for s in strList:
            res += s
        print(res)


if __name__ == '__main__':
    data_iter = get_iter()
    lstm = LSTMCore(config)
    generator = GeneratorModule(lstm, config)

    criterion = nn.MSELoss()
    gen_optimizer = optim.Adam(params=generator.parameters(), lr=config.generator_lr)

    train_generator_MLE(data_iter, criterion, gen_optimizer)
    generate_final_sentences(generator)