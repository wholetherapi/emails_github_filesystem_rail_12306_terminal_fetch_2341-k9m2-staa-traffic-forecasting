"""Sparse temporal attention encoder."""

import torch
import torch.nn as nn


class SparseTemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, adjacency):
        # x: (batch, nodes, time, channels)
        q = self.query(x).view(*x.shape[:3], self.heads, -1)
        k = self.key(x).view(*x.shape[:3], self.heads, -1)
        v = self.value(x).view(*x.shape[:3], self.heads, -1)
        scores = torch.einsum("bntch,bnshc->bnthc", q, k)
        # sparse mask: keep only neighbours present in the sensor graph
        mask = adjacency.unsqueeze(-1).unsqueeze(-1).bool()
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores / (self.hidden_dim ** 0.5), dim=1)
        context = torch.einsum("bnshc,bnthc->bnthc", self.dropout(weights), v)
        context = context.reshape(*x.shape[:3], self.hidden_dim)
        return x + self.out(context)
