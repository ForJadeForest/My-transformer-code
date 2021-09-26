from torch import nn
import torch
import numpy as np
import copy
import torch.functional as F
from torch.autograd import Variable


class Embedding(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        x = torch.tensor(x)
        return self.embedding(x)


class PositionEmbedding(nn.Module):
    def __init__(self, max_len, embedding_dim):
        super(PositionEmbedding, self).__init__()
        self.max_len = max_len
        self.embedding_dim = embedding_dim

    def forward(self, x):
        position = [
            [p / (1e4 ** (i / self.embedding_dim))
             for i in range(0, self.embedding_dim, 2)]
            for p in range(self.max_len)
        ]
        position = torch.tensor(position)
        position_embedding = torch.zeros(self.max_len, self.embedding_dim)
        position_embedding[:, 0::2] = torch.sin(position)
        position_embedding[:, 1::2] = torch.cos(position)
        position_embedding = position_embedding.unsqueeze(0)
        sentence_length = x.size(1)
        return x + Variable(position_embedding[:, sentence_length, :])


class MutiHeadAttention(nn.Module):
    def __init__(self, v_dim, k_dim, head_num, embedding_dim, nomal_factor):
        super(MutiHeadAttention, self).__init__()
        self.v_dim = v_dim
        self.k_dim = k_dim
        self.q_dim = k_dim
        self.heda_num = head_num
        self.nomal_factor = nomal_factor
        self.input_dim_tuple = (self.q_dim, self.k_dim, self.v_dim)
        self.linear_list = nn.ModuleList(
            [nn.Linear(embedding_dim, head_num * vec_dim) for vec_dim in self.input_dim_tuple])
        self.o_matrix = nn.Linear(self.v_dim * head_num, embedding_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, decoder_input=None):
        if decoder_input is None:
            decoder_input = x
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        input_tuple = (decoder_input, x, x)
        query, key, value = [
            linear_layer(layer_input).view(batch_size, seq_len, self.heda_num, input_dim).transpose(1, 2)
            for linear_layer, layer_input, input_dim
            in zip(self.linear_list, input_tuple, self.input_dim_tuple)
        ]

        score = torch.matmul(query, key.transpose(-2, -1)) * self.nomal_factor
        if mask is not None:
            score = score.masked_fill(mask, -float('inf'))
        score = self.softmax(score)
        output = torch.matmul(score, value).transpose(1, 2).reshape(batch_size, seq_len, self.heda_num * self.v_dim)
        output = self.o_matrix(output)
        return output
