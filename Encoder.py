from torch import nn
from MultiheadAttention import MultiheadAttentionBlock
from model import FeedForward
from model import LayerNormalization
from model import ResidualConnection


class EncoderBlock(nn.Module):
    def __init__(
        self,
        self_attention: MultiheadAttentionBlock,
        feed_forward: FeedForward,
        dropout: float,
    ):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention(x, x, x, src_mask)
        )
        x = self.residual_connections[1](x, self.feed_forward)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
