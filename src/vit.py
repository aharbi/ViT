import torch
import torch.nn as nn


class ViTProjectionBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 768):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.projection = nn.Linear(self.input_size, self.hidden_size)

    def forward(self, x):
        return self.projection(x)


class ViTEncoderBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        mlp_size: int = 3072,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.mlp_size = mlp_size
        self.dropout_rate = dropout_rate

        self.attention_norm = nn.LayerNorm(self.hidden_size)
        self.attention = nn.MultiheadAttention(self.hidden_size, self.num_heads)
        self.attention_dropout = nn.Dropout(self.dropout_rate)

        self.mlp_block_norm = nn.LayerNorm(self.hidden_size)
        self.mlp_block = nn.Sequential(
            nn.Linear(self.hidden_size, self.mlp_size),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.mlp_size, self.hidden_size),
            nn.Dropout(self.dropout_rate),
        )

    def forward(self, x):

        x_attention = self.attention_norm(x)
        x_attention, _ = self.attention(x_attention, x_attention, x_attention)

        x_attention = self.attention_dropout(x_attention)

        x = x + x_attention

        x_mlp = self.mlp_block_norm(x)
        x_mlp = self.mlp_block(x_mlp)

        x = x + x_mlp

        return x


class ViTClassificationHead(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 768):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_size = hidden_size

        self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation = nn.Tanh()
        self.linear_2 = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        # x[:, 0] is the classification token [CLS] (not working atm)
        # c = self.linear_1(x[:, 0])
        c = self.linear_1(x.mean(1))
        c = self.activation(c)
        c = self.linear_2(c)

        return c


class ViT(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_size: int = 3072,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_size = mlp_size
        self.dropout_rate = dropout_rate

        self.pos_embedding = nn.Parameter(
            torch.zeros(1, 3 + 1, self.hidden_size), requires_grad=True
        )

        self.projection_layer = ViTProjectionBlock(
            input_size=self.input_size, hidden_size=self.hidden_size
        )

        self.encoder_layers = [
            ViTEncoderBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_size=self.mlp_size,
                dropout_rate=self.dropout_rate,
            )
            for _ in range(self.num_layers)
        ]

        self.encoder = nn.Sequential(*self.encoder_layers)

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, self.hidden_size), requires_grad=True
        )

        self.classification_layer = ViTClassificationHead(
            num_classes=self.num_classes, hidden_size=self.hidden_size
        )

    def forward(self, x):

        x = self.projection_layer(x)

        # Append the learned classification token (Not working atm)
        # cls = self.cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls, x), dim=1)
        pos_embedding = self.pos_embedding.expand(x.shape[0], -1, -1)
        x = x + pos_embedding

        x = self.encoder(x)
        
        x = self.classification_layer(x)

        return x


if __name__ == "__main__":
    # Sanity check (ViT-Base)
    vit = ViT(
        input_size=49,
        num_classes=10,
        hidden_size=768,
        num_heads=12,
        num_layers=12,
        mlp_size=3072,
        dropout_rate=0.1,
    )

    x = torch.randn(32, 49, 49)

    y = vit(x)
