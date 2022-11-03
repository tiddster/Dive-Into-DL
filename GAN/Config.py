from DataProcess import word2id
class Config:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.hidden_dim = 100
        self.embedding_dim = 100

        self.generate_seq_len = 7
        self.generate_seq_num = 4
        self.n_rollout = 32
        self.generator_lr = 0.001

        self.epochs_nums = 500
        self.batch_size = 32

        self.num_filters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        self.filter_sizes = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]

        self.dropout = 0.1


config = Config(len(word2id))
