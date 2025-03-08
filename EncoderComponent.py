from torch import nn
from MultiheadAttention import MultiheadAttentionBlock
from layers import FeedForward, LayerNormalization, ResidualConnection

class EncoderBlock(nn.Module):
    def __init__(
        self,

        d_model: int,  # Added to match the call in build_transformer

        self_attention: MultiheadAttentionBlock,
        feed_forward: FeedForward,
        dropout: float,
    ):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(features,dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention(x, x, x, src_mask)
        )
        x = self.residual_connections[1](x, self.feed_forward)
        return x


class Encoder(nn.Module):

    
    def __init__(self, d_model: int, layers: nn.ModuleList) -> None:  # Added d_model parameter

        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)