"""
Fine-tuned autoencoder model for downstream tasks.
Inherits from the pretrained Autoencoder and adds a prediction head.
"""

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.autoencoder import Autoencoder
from src.utils.constants import MAX_CONTEXT_LENGTH


class AutoencoderFinetuned(Autoencoder):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        device: str,
        num_heads: int,
        num_layers: int,
        head_dim: int,
        prediction_dim: int,  # Final prediction dimension (e.g., 1 for regression)
        attention_hidden_dim: int,  # Hidden dimension for attention MLP
        mlp_hidden_dim: int,  # Hidden dimension for the final MLP
        dropout_rate: float,  # Dropout rate for regularization
        max_len: int = MAX_CONTEXT_LENGTH,
        name: str = "autoencoder_finetuned",
    ):
        super(AutoencoderFinetuned, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            device=device,
            num_heads=num_heads,
            num_layers=num_layers,
            head_dim=head_dim,
            max_len=max_len,
            name=name,
        )

        self.prediction_dim = prediction_dim
        hidden_dim = head_dim * num_heads

        # Simple MLP attention mechanism to aggregate across timesteps
        # Note: Uses input_dim (32) since super().forward() returns reconstructed output
        self.attention_mlp = nn.Sequential(
            nn.Linear(input_dim, attention_hidden_dim),
            nn.GELU(),
            nn.Linear(
                attention_hidden_dim, 1
            ),  # Output attention score for each timestep
        )

        # MLP prediction head with GELU activation
        # Note: Uses input_dim (32) since super().forward() returns reconstructed output
        self.prediction_head = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, prediction_dim),
        )

    def load_pretrained(self, pretrained_model: Autoencoder):
        """Load weights from a pretrained Autoencoder model, excluding the prediction head."""

        if self.input_dim != pretrained_model.input_dim:
            raise ValueError(
                f"expected input dimension {self.input_dim} but received {pretrained_model.input_dim}"
            )
        if self.max_len != pretrained_model.max_len:
            raise ValueError(
                f"expected max length {self.max_len} but received {pretrained_model.max_len}"
            )

        # Load pretrained encoder components
        self.in_proj = copy.deepcopy(pretrained_model.in_proj)
        self.positional_encoding = copy.deepcopy(pretrained_model.positional_encoding)
        self.transformer_encoder = copy.deepcopy(pretrained_model.transformer_encoder)

        # Note: We don't load the out_proj from pretrained model since we have our own prediction head
        # The pretrained out_proj was for reconstruction, our prediction head is for classification/regression

    def forward(
        self,
        input_tensor: torch.Tensor,
        input_feature_mask: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for fine-tuning.

        Args:
            input_tensor: batch_size x seq_len x n_features
            input_feature_mask: batch_size x seq_len x n_features (optional for fine-tuning)
            src_key_padding_mask: batch_size x seq_len

        Returns:
            predictions: batch_size x prediction_dim
        """
        encoder_output = super().forward(
            input_tensor, input_feature_mask, src_key_padding_mask
        )
        # encoder_output: batch_size x seq_len x hidden_dim

        # Use simple MLP attention to aggregate across timesteps
        attention_scores = self.attention_mlp(
            encoder_output
        )  # batch_size x seq_len x 1
        attention_weights = F.softmax(
            attention_scores, dim=1
        )  # batch_size x seq_len x 1
        compressed_representation = (encoder_output * attention_weights).sum(
            dim=1
        )  # batch_size x hidden_dim
        predictions = self.prediction_head(
            compressed_representation
        )  # batch_size x prediction_dim

        return predictions
