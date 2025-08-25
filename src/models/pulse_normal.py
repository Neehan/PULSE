import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.models.autoencoder import Autoencoder
from src.utils.constants import MAX_CONTEXT_LENGTH, VAR_MIN, VAR_MAX

"""
This class implements the PULSENormal model.

The output is a tuple of mu, sigma, where mu is the mean and sigma is the 
standard deviation of the normal distribution.
"""


class PULSENormal(Autoencoder):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        device: str,
        num_heads: int,
        num_layers: int,
        head_dim: int,
        max_len: int = MAX_CONTEXT_LENGTH,
        name: str = "pulsenormal",
    ):
        super(PULSENormal, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            head_dim=head_dim,
            max_len=max_len,
            device=device,
            name=name,
        )
        # Override the output projection to be 2x the size for mu and sigma
        hidden_dim = head_dim * num_heads
        self.out_proj = nn.Linear(hidden_dim, 2 * output_dim)

    def load_pretrained(
        self,
        pretrained_model: "PULSENormal",
        load_out_proj: bool = True,
    ):
        """Load weights from a pretrained WeatherBERT model by deep copying each layer."""
        if isinstance(pretrained_model, PULSENormal):
            super().load_pretrained(pretrained_model)
        else:
            raise ValueError(
                f"Expected pretrained model class to be PULSENormal, but got {type(pretrained_model)}"
            )

    def forward(
        self,
        input_tensor: torch.Tensor,
        input_feature_mask: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        input_tensor: batch_size x seq_len x n_features
        input_feature_mask: batch_size x seq_len x n_features
        src_key_padding_mask: batch_size x seq_len
        """
        # Call parent Autoencoder forward method
        output = super().forward(
            input_tensor=input_tensor,
            input_feature_mask=input_feature_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

        # Split output into mu and log_var (VAE-style parameterization)
        mu_x = output[..., : self.output_dim]
        log_var_x = output[..., self.output_dim :]
        var_x = torch.exp(log_var_x)

        # Clip sigma to prevent numerical instability and overly negative log terms
        var_x = torch.clamp(var_x, min=VAR_MIN, max=VAR_MAX)  # sigma is in [0.0001, 1]

        return mu_x, var_x
