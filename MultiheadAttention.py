import torch
import torch.nn as nn
import math


class MultiheadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model isn't divisible by h"

        self.d_k = d_model // h
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)  # it's dimensions is equivaent to (h*d_k, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):

        d_k = query.shape[-1]
        # (batch_size, h, seq_len, d_k) @ (batch_size, h, d_k, seq_len) --> (batch_size, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):

        print(f"q shape: {q.shape}")  # Debug print statement
        print(f"k shape: {k.shape}")  # Debug print statement
        print(f"v shape: {v.shape}")  # Debug print statement

        # if q.dim() == 2:
        #     q = q.unsqueeze(-1).expand(-1, -1, self.d_model)
        # if k.dim() == 2:
        #     k = k.unsqueeze(-1).expand(-1, -1, self.d_model)
        # if v.dim() == 2:
        #     v = v.unsqueeze(-1).expand(-1, -1, self.d_model)

        query = self.W_q(q)  # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        key = self.W_k(k)
        value = self.W_v(v)

        print(f"query shape after W_q: {query.shape}")  # Debug print statement
        print(f"key shape after W_k: {key.shape}")  # Debug print statement
        print(f"value shape after W_v: {value.shape}")  # Debug print statement


        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, h, d_k) --> (batch_size, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiheadAttentionBlock.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)  # (batch_size, h, seq_len, d_k) --> (batch_size, seq_len, h, d_k) --> (batch_size, seq_len, d_model)

        return self.W_o(x)
