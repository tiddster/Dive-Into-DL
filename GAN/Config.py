from DataProcess import word2id,max_seqLen
class Config:
    def __init__(self, vocab_size, max_seqLen):
        self.vocab_size = vocab_size + 1
        self.hidden_dim = 100
        self.embedding_dim = 100

        self.generate_seq_len = 30
        self.n_rollout = 1
        self.generator_lr = 0.0001

        self. max_seqLen = max_seqLen

        self.epochs_nums = 10
        self.batch_size = 32

        self.update_rater = 0.8

        self.num_filters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        self.filter_sizes = [3, 5, 5, 5, 5, 3, 3, 3, 3, 3, 5, 5]

        self.dropout = 0.1


config = Config(len(word2id), max_seqLen)
