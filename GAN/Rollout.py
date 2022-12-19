import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


class Rollout(object):
    """ Rollout Policy """

    def __init__(self, model, update_rate):
        self.ori_model = model
        self.own_model = copy.deepcopy(model)
        self.update_rate = update_rate

    def get_reward(self, x, num, discriminator):
        """
        Inputs: x, num, discriminator
            - x: (batch_size, seq_len) input data
            - num: rollout number
            - discriminator: discrimanator model
        """
        rewards = []
        batch_size = x.size(0)
        seq_len = x.size(1)
        for i in range(num):
            for l in range(1, seq_len):
                data = x[:, 0:l]
                samples = self.own_model.generate(data, batch_size, seq_len)
                pred = discriminator(samples)
                pred = pred.cpu().data[:,1].numpy()
                if i == 0:
                    rewards.append(pred)
                else:
                    rewards[l-1] += pred
                # print(f"{i}    {l}   {pred}   {rewards}")

            # 计算最后一个字的reward
            pred = discriminator(x)
            pred = pred.cpu().data[:, 1].numpy()
            if i == 0:
                rewards.append(pred)
            else:
                rewards[seq_len-1] += pred
            # print(f"{i}    {pred}   {rewards}")
        rewards = np.transpose(np.array(rewards)) / (1.0 * num) # batch_size * seq_len
        # print(rewards)
        return rewards

    def get_reward2(self, x, num, discriminator):
        """
        Inputs: x, num, discriminator
            - x: (batch_size, seq_len) input data
            - num: rollout number
            - discriminator: discrimanator model
        """
        rewards = []
        batch_size = x.size(0)
        seq_len = x.size(1)
        for i in range(num):
            for l in range(1, seq_len):
                data = x[:, 0:l]
                samples = self.own_model.generate(data, batch_size, seq_len)
                pred = discriminator(samples)
                pred = pred.cpu().data[:,1].numpy()
                if i == 0:
                    rewards.append(pred)
                else:
                    if pred.sum() > rewards[l-1].sum():
                        rewards[l-1] = pred
                        x[:, l] = samples[:, l]
                # print(f"{i}    {l}   {pred}   {rewards}")

            # 计算最后一个字的reward
            pred = discriminator(samples)
            pred = pred.cpu().data[:, 1].numpy()

            if i == 0:
                rewards.append(pred)
            else:
                if pred.sum() > rewards[seq_len-1].sum():
                    rewards[seq_len - 1] = pred
                    x[:, seq_len-1] = samples[:, seq_len-1]
            # print(f"{i}    {pred}   {rewards}")
        # print(rewards)
        return rewards, x

    def update_params(self):
        dic = {}
        for name, param in self.ori_model.named_parameters():
            dic[name] = param.data
        for name, param in self.own_model.named_parameters():
            if name.startswith('emb'):
                param.data = dic[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]