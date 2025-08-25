"""
This class implements a simple autoencoder model with a transformer encoder.
"""

import copy
from typing import Optional

import torch
import torch.nn as nn

from src.base.base_model import BaseModel
from src.models.vanilla_pos_encoding import VanillaPositionalEncoding
from src.utils.constants import MAX_CONTEXT_LENGTH


class Autoencoder(BaseModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        device: str,
        num_heads: int,
        num_layers: int,
        head_dim: int,
        max_len: int = MAX_CONTEXT_LENGTH,
        name: str = "autoencoder",
    ):
        super(Autoencoder, self).__init__(name)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_len = max_len

        hidden_dim = head_dim * num_heads
        feedforward_dim = hidden_dim * 4

        self.in_proj = nn.Linear(self.input_dim, hidden_dim)

        self.positional_encoding = VanillaPositionalEncoding(
            hidden_dim, max_len=max_len, device=device
        )
        encoder_layer = nn.TransformerEncoderLayer(
            batch_first=True,
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            device=device,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def load_pretrained(self, pretrained_model: "Autoencoder"):
        """Load weights from a pretrained WeatherBERT model by deep copying each layer."""

        if self.input_dim != pretrained_model.input_dim:
            raise ValueError(
                f"expected input dimension {self.input_dim} but received {pretrained_model.input_dim}"
            )
        if self.max_len != pretrained_model.max_len:
            raise ValueError(
                f"expected max length {self.max_len} but received {pretrained_model.max_len}"
            )

        self.in_proj = copy.deepcopy(pretrained_model.in_proj)
        self.positional_encoding = copy.deepcopy(pretrained_model.positional_encoding)
        self.transformer_encoder = copy.deepcopy(pretrained_model.transformer_encoder)
        self.out_proj = copy.deepcopy(pretrained_model.out_proj)

    def forward(
        self,
        input_tensor: torch.Tensor,
        input_feature_mask: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        input_tensor: batch_size x seq_len x n_features
        input_feature_mask: batch_size x seq_len x n_features
        src_key_padding_mask: batch_size x seq_len
        """
        # mask input_tensor for the masked dimensions
        input_tensor = input_tensor * (~input_feature_mask)

        input_tensor = self.in_proj(input_tensor)
        input_tensor = self.positional_encoding(input_tensor)
        input_tensor = self.transformer_encoder(
            input_tensor, src_key_padding_mask=src_key_padding_mask
        )
        output = self.out_proj(input_tensor)

        return output
