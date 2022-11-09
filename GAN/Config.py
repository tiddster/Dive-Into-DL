from DataProcess import word2id,max_seqLen

class Config:
    def __init__(self, vocab_size, max_seqLen):
        self.vocab_size = vocab_size
        self.hidden_dim = 100
        self.embedding_dim = 100

        self.n_rollout = 2

        self.generate_seq_len = max_seqLen

        self.generator_nll_lr = 0.00005
        self.generator_pg_lr = 0.001

        self. max_seqLen = max_seqLen

        self.epochs_nums = 5
        self.batch_size = 64

        self.update_rater = 0.8

        self.num_filters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        self.filter_sizes = [3, 5, 5, 5, 5, 3, 3, 3, 3, 3, 5, 5]

        self.dropout = 0.1


config = Config(len(word2id), max_seqLen)
