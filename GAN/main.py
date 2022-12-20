import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from matplotlib import pyplot as plt

from DataProcess import get_iter, pos_path, neg_path, id2word, word2id
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
    loss_list = []
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
    loss_list.append(avg_loss)
    print(f"G---------train loss: {avg_loss:.8f}")
    return avg_loss


def train_discriminator(data_iter, criterion, optimizer):
    g_win, d_acc = 0, 0
    total_loss, total, g_total = 0.0, 0, 0
    for data, labels in data_iter:
        data, labels = data.to(device), labels.to(device)
        labels = labels.contiguous().view(-1)
        output = discriminator(data)
        loss = criterion(output, labels)
        loss.requires_grad_(True)
        total_loss += loss.item()

        for l, o in zip(labels, output):
            pred = o.argmax()
            if l == pred:
                d_acc += 1
            if l.item() == 0:
                g_total += 1
                if pred.item() == 1:
                    g_win += 1

        total += labels.shape[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(data_iter)
    print(
        f"D---------train loss: {avg_loss:.8f}, d_acc: {d_acc / total * 100:2f}%, g_win: {g_win / g_total * 100:0.2f}%")
    return avg_loss, g_win / g_total * 100, d_acc / total * 100


def train_generator_PG(gen, dis, rollout, pg_loss, optimizer):
    """
    1. 先生成batch_size个句子序列
    2. 将以上句子序列，从0到末尾以此放入鉴别器中，算出reward平均值
    3. 利用PG算法算出损失值
    """
    # construct the input to the genrator, add zeros before samples and delete the last column
    samples = generator.generate(generate_name=config.generate_num, seq_len=config.generate_seq_len)
    # zeros = torch.zeros(config.batch_size, 1, dtype=torch.int64)
    # zeros = zeros.to(device)

    inputs = samples
    targets = samples

    # calculate the reward
    Qvalue, x = rollout.get_reward2(samples, config.n_rollout, dis)
    Qvalue = torch.tensor(Qvalue)
    Qvalue = Qvalue.transpose(0, 1)
    Qvalue = Qvalue.to(device)

    sample_num, samples_len = Qvalue.shape
    Qvalue_sum = Qvalue.sum(dim=1)
    Qvalue_sum = Qvalue_sum.sum().item()
    Q_avg = Qvalue_sum / sample_num

    # update generator
    output = gen(inputs)
    rewards_loss = pg_loss(output, targets, Qvalue)

    optimizer.zero_grad()
    rewards_loss.backward()
    optimizer.step()
    print(f"GD--------- train loss: {rewards_loss:.5f}, rewards: {Q_avg:.4f}")

    return rewards_loss.item(), Q_avg


# 打印最终的完整句段和句子所对应的token
def generate_final_sentences(generator, generate_num=config.generate_num, tokens=None):
    token_samples = generator.generate(tokens, generate_num)
    texts = ""
    for tokens in token_samples:
        for token in tokens:
            texts += id2word[token]
        texts += '\n'

    # print(texts)

    with open(neg_path, 'w', encoding='utf8') as f:
        f.write(texts)


def generate_with_hint(generator, hints):
    idList = [[word2id[h]] for h in hints]
    idList = torch.tensor(idList).int()
    print(idList)
    generate_final_sentences(generator, 4, idList)


def polt_losses(rewardsList, QList, GLossList, DLossList, GwinList, DAccList):
    x = range(0, len(rewardsList), 1)

    plt.plot(x, rewardsList)
    plt.xlabel("epochs")
    plt.ylabel("策略梯度函数损失值")
    plt.show()

    plt.plot(x, QList)
    plt.xlabel("epochs")
    plt.ylabel("价值组Q平均值")
    plt.show()

    plt.plot(x, GLossList)
    plt.xlabel("epochs")
    plt.ylabel("生成器训练损失值")
    plt.show()

    plt.plot(x, DLossList)
    plt.xlabel("epochs")
    plt.ylabel("鉴定器训练损失值")
    plt.show()

    plt.plot(x, GwinList)
    plt.xlabel("epochs")
    plt.ylabel("生成器“欺骗”成功率")
    plt.show()

    plt.plot(x, DAccList)
    plt.xlabel("epochs")
    plt.ylabel("鉴定器鉴定正确率")
    plt.show()


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']

    print("读取数据中")
    pos_data_iter = get_iter()
    print("读取数据完成")

    lstm = LSTMCore(config)
    generator = GeneratorModule(lstm, config)
    discriminator = DiscriminatorModule(config)
    rollout = Rollout(generator, config.update_rater)

    pg_criterion = PGLoss()
    nll_criterion = nn.NLLLoss()
    cel_criterion = nn.CrossEntropyLoss()
    nll_optimizer = optim.Adam(params=generator.parameters(), lr=config.generator_nll_lr)
    pg_optimizer = optim.Adam(params=generator.parameters(), lr=config.generator_pg_lr)
    discriminator_optimizer = optim.Adam(params=discriminator.parameters(), lr=config.generator_nll_lr)

    _ = train_generator_NLL(pos_data_iter, nll_criterion, nll_optimizer)
    generate_final_sentences(generator)
    all_data_iter = get_iter(pos_path, neg_path)
    _ = train_discriminator(all_data_iter, cel_criterion, discriminator_optimizer)


    rewardsLossList = []
    QList = []
    GLossList = []
    DLossList = []
    GwinList = []
    DAccList = []
    for i in range(100):
        start = time.time()
        GLoss = train_generator_NLL(pos_data_iter, nll_criterion, nll_optimizer)
        GLossList.append(GLoss)

        rewardsLoss, QValue = train_generator_PG(generator, discriminator, rollout, pg_criterion, pg_optimizer)
        rewardsLossList.append(-rewardsLoss)
        QList.append(QValue)

        generate_final_sentences(generator)

        all_data_iter = get_iter(pos_path, neg_path)
        DLoss, g_win, d_acc = train_discriminator(all_data_iter, cel_criterion, discriminator_optimizer)
        DLossList.append(DLoss)
        GwinList.append(g_win)
        DAccList.append(d_acc)

        end = time.time()
        print(f"epoch: {i}, time: {end - start}")

    generate_with_hint(generator, "机器学习")
    polt_losses(rewardsLossList, QList, GLossList, DLossList, GwinList, DAccList)
