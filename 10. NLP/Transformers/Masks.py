import torch


def padding_mask(seq_Q, seq_K):
    '''
    seq_Q: [batch_size, seq_len]
    seq_K: [batch_size, seq_len]
    '''
    batch_size, len_Q = seq_Q.shape
    batch_size, len_K = seq_K.shape
    # 判断是否为0，若不为0则是false，不mask；若为0则True，需要被mask
    pad_attn_mask = seq_K.eq(0).unsqueeze(1)
    return pad_attn_mask.expand((batch_size, len_Q, len_K))  # [batch_size, len_q, len_k]


def sequence_mask(seq):
    """
    是对于一个序列，在time_step为t的时刻，我们的解码输出应该只能依赖于t时刻之前的输出，而不能依赖t之后的输出。
    因此我们需要想一个办法，把t之后的信息给隐藏起来。
    所以创造一个上三角矩阵
    :param seq:
    :return:
    """
    batch_size, seq_len = seq.shape
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8), diagonal=1)
    mask = mask.expand((batch_size, seq_len, seq_len))
    return mask
