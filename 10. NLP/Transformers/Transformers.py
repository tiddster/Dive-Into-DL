import torch
import torch.nn as nn

from EncoderAndDecoder import Encoder, Decoder

class Config():
    def __init__(self):
        self.vocab_size = 10
        self.max_len = 500
        self.model_dim = 512
        self.head_dim = 512


class Transformers(nn.Module):
    def __init__(self, config):
        super(Transformers, self).__init__()
        self.encoder = Encoder(config.vocab_size, config.model_dim, config.head_dim, config.max_len)
        self.decoder = Decoder(config.vocab_size, config.model_dim, config.head_dim, config.max_len)
        self.fc = nn.Linear(config.model_dim, config.vocab_size)

    def forward(self, encoder_inputs, decoder_inputs):
        """
        :param encoder_inputs:  [batch_size, seq_len]
        :param decoder_inputs:  [batch_size, seq_len]
        :return:
        """
        # [batch_size, seq_len] -> [batch_size, seq_len, model_dim]
        encoder_outputs = self.encoder(encoder_inputs)

        # [batch_size, seq_len] -> [batch_size, seq_len, model_dim]
        decoder_outputs = self.decoder(decoder_inputs, encoder_outputs)

        final_output = self.fc(decoder_outputs)
        return final_output.view(-1, final_output.shape[-1])


text = ["I like basketball", "I hate you", "I am sorry", "You love me"]
textSeq = torch.tensor([[1, 2, 3, 0, 0], [1, 4, 5, 0, 0], [1, 6, 7, 0, 0], [5, 8, 9, 0, 0]]).long()
print(textSeq.shape)

config = Config()
t = Transformers(config)
print(t(textSeq, textSeq).shape)