path = "P:\Dataset\GAN\\ChinesePoem.txt"

def get_data(path):
    with open(path, 'r', encoding='utf8') as f:
        textList = []
        for text in f.readlines():
            text = text.replace('\n', '').replace('\ufeff', '')
            textList.append(text)
        return textList


def get_vocab(textList):
    id2word  = []
    for text in textList:
        texts = list(set(text))
        id2word += texts

    id2word = list(set(id2word))
    word2id = {word: i for i, word in enumerate(id2word)}
    return word2id


def get_token(textList, word2id):
    tokenList = []
    for text in textList:
        tokens = []
        for t in text:
            tokens.append(word2id(t))
        tokenList.append(tokens)

    return tokenList
