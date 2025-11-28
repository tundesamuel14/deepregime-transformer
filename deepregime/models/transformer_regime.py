import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Adds a sense of "time step" into the embeddings so the model knows order.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class RegimeTransformer(nn.Module):
    """
    Input:  (batch, seq_len, num_features)
    Output: (batch, num_regimes) - logits for each regime
    """

    def __init__(
        self,
        num_features: int,
        num_regimes: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Project raw features into d_model space
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (batch, seq, feature)
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, num_regimes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, num_features)
        """
        x = self.input_proj(x)           # (batch, seq_len, d_model)
        x = self.pos_encoder(x)          # add time info
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)

        last_token = x[:, -1, :]         # representation of last time step
        logits = self.fc_out(self.dropout(last_token))
        return logits
