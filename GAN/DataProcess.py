from torch.utils.data import Dataset, DataLoader
import torch

path = "P:\Dataset\GAN\\ChinesePoem.txt"


def get_data(path):
    with open(path, 'r', encoding='utf8') as f:
        textList = []
        for text in f.readlines():
            text = text.replace('\n', '').replace('\ufeff', '')
            textList.append(text)
        return textList


def get_vocab(textList):
    id2word = []
    for text in textList:
        texts = list(set(text))
        id2word += texts

    id2word = list(set(id2word))
    word2id = {word: i for i, word in enumerate(id2word)}
    return id2word, word2id


def get_token(textList, word2id):
    tokenList = []
    for text in textList:
        tokens = []
        for t in text:
            tokens.append(word2id[t])
        tokenList.append(tokens)
    return tokenList


id2word, word2id = get_vocab(get_data(path))


def get_iter(is_positive=True, path=path):
    textList = get_data(path)
    if is_positive:
        tokenList = get_token(textList, word2id)
        labels = torch.ones(len(tokenList))
    else:
        tokenList = get_token(textList, word2id)
        labels = torch.zeros(len(tokenList))
    print(tokenList)
    dataset = GANDataset(tokenList, labels)
    iter = DataLoader(dataset, batch_size=32, shuffle=True)
    return iter


class GANDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x)
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.y.shape[0]
