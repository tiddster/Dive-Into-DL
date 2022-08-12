import random
import torch

path = "F:\MINSTDataset\Lyrics\jaychou_lyrics.txt"

with open(path, encoding='utf-8') as f:
    s = f.read()

"""
创造两个列表：
1、 通过索引寻找字符
2、 通过字符寻找索引
"""
idx_to_char = list(set(s))
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
vocab_size = len(char_to_idx)

"""
将歌词转化成索引
"""
indices = [char_to_idx[char] for char in s]

"""
隔空采样
"""
def data_iter_random(lyrics_indices, batch_size, num_steps, device=None):
    num_examples = (len(lyrics_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    def _data(pos):
        return lyrics_indices[pos : pos + num_steps]

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(epoch_size):
        i *= batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(Y, dtype=torch.float32, device=device)

# my_seq = list(range(30))
# for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=6):
#     print('X: ', X, '\nY:', Y, '\n')

def data_iter_consecutive(lyrics_indices, batch_size, num_steps, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lyrics_indices = torch.tensor(lyrics_indices, dtype=torch.float32, device=device)
    data_len = len(lyrics_indices)
    batch_len = data_len // batch_size
    indices = lyrics_indices[0: batch_size*batch_len].view(batch_size, batch_len)
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y

# for X, Y in data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
#     print('X: ', X, '\nY:', Y, '\n')


def load_data_jay_lyrics():
    return indices, char_to_idx, idx_to_char, vocab_size