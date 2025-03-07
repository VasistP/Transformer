import torch
import torch.nn as nn
import math
from MultiheadAttention import MultiheadAttentionBlock
from EncoderComponent import Encoder, EncoderBlock
from DecoderComponent import Decoder, DecoderBlock
from layers import FeedForward


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

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # pe = positional encoding
        pe = torch.zeros(seq_len, d_model)
        # PE(pos, 2i) = sin(pos/10000^(2i/d_model)) for even positions
        # PE(pos, 2i+1) = cos(pos/10000^(2i/d_model)) for odd positions

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        self.register_buffer("pe", pe)  # Registering as buffer to save it in the model's state_dict. This is used to save the model and load it later

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.projection(x)
        # return torch.log_softmax(self.projection(x), dim=-1)


class Transformer(nn.Module):

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbeddings,
        target_embedding: InputEmbeddings,
        src_position: PositionalEncoding,
        target_position: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.target_embedding = target_embedding
        self.src_position = src_position
        self.target_position = target_position
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_position(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor):
        target = self.target_embedding(target)
        target = self.target_position(target)
        return self.decoder(target, encoder_output, src_mask, target_mask)

    def project(self, x):
        return self.projection_layer(x)


def build_transformer(
    src_vocab_size: int,
    target_vocab_size: int,
    src_seq_len: int,
    target_seq_len: int,
    d_model: int = 512,
    d_ff: int = 2048,
    num_heads: int = 8,
    num_layers: int = 6,
    dropout: float = 0.1,
) -> Transformer:
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    target_embed = InputEmbeddings(d_model, target_vocab_size)

    src_position = PositionalEncoding(d_model, src_seq_len, dropout)
    target_position = PositionalEncoding(d_model, target_seq_len, dropout)

    encoder_blocks = []
    for _ in range(num_layers):
        encoder_self_attention = MultiheadAttentionBlock(d_model, num_heads, dropout)
        encoder_feed_forward = FeedForward(d_model, d_ff, dropout)
        encoder_blocks.append(
            # Encoder.
            EncoderBlock(d_model, encoder_self_attention, encoder_feed_forward, dropout)
        )

    decoder_blocks = []
    for _ in range(num_layers):
        decoder_self_attention = MultiheadAttentionBlock(d_model, num_heads, dropout)
        decoder_cross_attention = MultiheadAttentionBlock(d_model, num_heads, dropout)
        decoder_feed_forward = FeedForward(d_model, d_ff, dropout)
        decoder_blocks.append(
            # Decoder.
            DecoderBlock(d_model, decoder_self_attention,decoder_cross_attention, decoder_feed_forward, dropout)
        )

    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model, target_vocab_size)

    transformer = Transformer(
        encoder,
        decoder,
        src_embed,
        target_embed,
        src_position,
        target_position,
        projection_layer,
    )

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
