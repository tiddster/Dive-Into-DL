class Config:
    def __init__(self):
        self.vocab_size = 100
        self.hidden_dim = 100
        self.embedding_dim = 100

        self.num_filters = [1, 1, 1]
        self.filter_sizes = [3, 3, 3]

        self.dropout = 0.1


config = Config()
