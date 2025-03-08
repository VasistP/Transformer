import torch
import torch.nn as nn
import math
from MultiheadAttention import MultiheadAttentionBlock
from layers import FeedForward, LayerNormalization, ResidualConnection


class DecoderBlock(nn.Module):

    def __init__(
        self,
        features: int,
        self_attention: MultiheadAttentionBlock,
        cross_attention: MultiheadAttentionBlock,
        feed_forward: FeedForward,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward

        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features,dropout) for _ in range(3)]
        )

    def forward(self, x, encoder_out, src_mask, target_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention(x, x, x, target_mask)
        )
        x = self.residual_connections[1](
            x, lambda x: self.cross_attention(x, encoder_out, encoder_out, src_mask)
        )
        x = self.residual_connections[2](x, self.feed_forward)
        return x


class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_out, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_out, src_mask, target_mask)
        return self.norm(x)
