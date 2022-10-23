import torch
import torch.nn as nn

from Attention import MultiHeadAttention
from FeedForwrad import FeedForwardNet
from Embeddings import PositionEmbeddings
from Masks import padding_mask, sequence_mask


# 残差连接和归一化在各个模块内部已经实现
class EncoderLayer(nn.Module):
    def __init__(self,head_dim, model_dim=512):
        super(EncoderLayer, self).__init__()
        self.mhLayer = MultiHeadAttention(head_dim)
        self.ffNet = FeedForwardNet(model_dim)

    def forward(self, inputs, mask=None):
        # [batch_size, seq_len, model_dim]
        output = self.mhLayer(inputs, inputs, inputs, mask)

        # [batch_size, seq_len, model_dim]
        output = self.ffNet(output)
        return output


class Encoder(nn.Module):
    def __init__(self,vocab_size, model_dim, head_dim, max_len=500):
        super(Encoder, self).__init__()
        self.vocabEmb = nn.Embedding(vocab_size, model_dim)
        self.posEmb = PositionEmbeddings(max_len, model_dim)

        self.EncoderLayers = nn.ModuleList([EncoderLayer(head_dim) for _ in range(6)])

    def forward(self, inputs):
        encoder_inputs_self_mask = padding_mask(inputs, inputs)

        # [batch_size, seq_len] -> [batch_size, seq_len, model_dim]
        inputs = self.vocabEmb(inputs)

        # [batch_size, seq_len, model_dim]
        inputs = self.posEmb(inputs)

        # [batch_size, seq_len, model_dim]
        outputs = inputs
        for layer in self.EncoderLayers:
            outputs = layer(outputs, encoder_inputs_self_mask)
        return outputs


class DecoderLayer(nn.Module):
    def __init__(self, head_dim, model_dim=512):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(head_dim)
        self.decoder_encoder_attention = MultiHeadAttention(head_dim)
        self.ffNet = FeedForwardNet(model_dim)

    def forward(self, decoder_input, encoder_outputs, self_mask=None, encoder_mask=None):
        outputs = self.self_attention(decoder_input, decoder_input, decoder_input, self_mask)
        outputs = self.decoder_encoder_attention(outputs, encoder_outputs, encoder_outputs, encoder_mask)
        output = self.ffNet(outputs)
        return output


class Decoder(nn.Module):
    def __init__(self, vocab_size, model_dim, head_dim, max_len):
        super(Decoder, self).__init__()
        self.vocabEmb = nn.Embedding(vocab_size, model_dim)
        self.posEmb = PositionEmbeddings(max_len, model_dim)
        self.DecoderLayers = nn.ModuleList([DecoderLayer(head_dim) for _ in range(6)])

    def forward(self, decoder_inputs, encoder_outputs):

        # [batch_size, seq_len] -> [batch_size, seq_len, model_dim]
        decoder_outputs = self.vocabEmb(decoder_inputs)
        decoder_outputs = self.posEmb(decoder_outputs)

        decoder_inputs_self_mask = padding_mask(decoder_inputs, decoder_inputs)
        seq_mask = sequence_mask(decoder_inputs)

        for layer in self.DecoderLayers:
            decoder_outputs = layer(decoder_outputs, encoder_outputs, decoder_inputs_self_mask, seq_mask)

        return decoder_outputs