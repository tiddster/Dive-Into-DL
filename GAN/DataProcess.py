from torch.utils.data import Dataset, DataLoader
import torch

pos_path = "P:\Dataset\GAN\\ChinesePoem.txt"
neg_path = "Dataset\\output.txt"

max_seqLen = 7

def get_data(path):
    with open(path, 'r', encoding='utf8') as f:
        textList = []
        for text in f.readlines():
            text = text.replace('\n', '').replace('\ufeff', '').replace('1','').replace(' ','').replace('-','')
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


id2word, word2id = get_vocab(get_data(pos_path))

def get_iter(pos_path=pos_path, neg_path=None):
    if neg_path:
        neg_text_list = get_data(neg_path)
        neg_labels = [0 for _ in range(len(neg_text_list))]
    else:
        neg_text_list = []
        neg_labels = []

    pos_text_list = get_data(pos_path)
    pos_labels = [1 for _ in range(len(pos_text_list))]

    textList = pos_text_list + neg_text_list
    labels = pos_labels + neg_labels

    tokenList = get_token(textList, word2id)
    for i in range(len(tokenList)):
        if len(tokenList[i]) > max_seqLen:
            tokenList[i] = tokenList[i][:max_seqLen]
        else:
            tokenList[i] = tokenList[i] + [0] *( max_seqLen - len(tokenList[i]))

    # finalList = []
    # for tokens in tokenList:
    #     for t in tokens:
    #         finalList.append(t)

    dataset = GANDataset(tokenList, labels)
    iter = DataLoader(dataset, batch_size=32, shuffle=True)
    return iter


class GANDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x).long()
        self.y = torch.tensor(y).long()

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.y.shape[0]
