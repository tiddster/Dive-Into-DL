from DataProcess import word2id
class Config:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size-1
        self.hidden_dim = 100
        self.embedding_dim = 100

        self.generate_seq_len = 30
        self.n_rollout = 32
        self.generator_lr = 0.001

        self.epochs_nums = 500
        self.batch_size = 32

        self.update_rater = 0.8

        self.num_filters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        self.filter_sizes = [3, 5, 5, 5, 5, 3, 3, 3, 3, 3, 5, 5]

        self.dropout = 0.1


config = Config(len(word2id))
