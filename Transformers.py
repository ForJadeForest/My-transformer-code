from torch import nn
import torch
import numpy as np


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

    def forward(self):
        position_embedding = [
            [position / (1e5 ** (2 * i / self.embedding_dim)) for i in range(self.embedding_dim)]
            for position in self.max_len
        ]
        position_embedding = torch.tensor(position_embedding)
        return position_embedding


class MutiHeadAttention(nn.Module):
    def __init__(self, v_dim, k_dim, q_dim, head_num, embedding_dim):
        super(MutiHeadAttention, self).__init__()
        self.v_dim = v_dim
        self.k_dim = k_dim
        self.q_dim = k_dim
        self.heda_num = head_num

        self.v_matrix = nn.Linear(embedding_dim, self.v_dim)
        self.k_matrix = nn.Linear(embedding_dim, self.k_dim)
        self.q_matrix = nn.Linear(embedding_dim, self.q_dim)



