import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(
            vocab_size, d_model
        )  # Replace with custom word embedding fucntion later

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # pe = positional encoding
        pe = torch.zeros(seq_len, d_model)
        # PE(pos, 2i) = sin(pos/10000^(2i/d_model)) for even positions
        # PE(pos, 2i+1) = cos(pos/10000^(2i/d_model)) for odd positions

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(
            1
        )  # (seq_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        self.register_buffer(
            "pe", pe
        )  # Registering as buffer to save it in the model's state_dict. This is used to save the model and load it later

    def forward(self, x):
        x = x + (self.pe[:, : x.size(1)]).requires_grad_(False)
        return self.dropout(x)
