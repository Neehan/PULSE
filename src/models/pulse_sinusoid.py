import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.models.pulse_normal import PULSENormal
from src.utils.constants import MAX_CONTEXT_LENGTH, VAR_MIN, VAR_MAX

"""
This class implements the PULSESinusoid model, which extends PULSENormal
with k sinusoidal prior components with parameters:
    mu_x: mean of the variational distribution
    var_x: variance of the variational distribution
    amplitude_k: amplitude of the k-th sinusoidal prior component
    phase_k: phase of the k-th sinusoidal prior component
    frequency_k: frequency of the k-th sinusoidal prior component
"""


class PULSESinusoid(PULSENormal):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        device: str,
        num_heads: int,
        num_layers: int,
        head_dim: int,
        k_components: int,
        max_len: int = MAX_CONTEXT_LENGTH,
        name: str = "pulsesinusoid",
    ):
        super(PULSESinusoid, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            device=device,
            num_heads=num_heads,
            num_layers=num_layers,
            head_dim=head_dim,
            max_len=max_len,
            name=name,
        )
        self.k = k_components
        self.positions = torch.arange(
            max_len, dtype=torch.float, device=device
        ).reshape(1, 1, max_len, 1)
        # Initialize with shape (1, k, max_len, input_dim) to avoid unsqueezing later
        self.frequency = nn.Parameter(torch.randn(1, self.k, max_len, input_dim) * 0.1)
        self.phase = nn.Parameter(torch.randn(1, self.k, max_len, input_dim) * 0.1)
        self.amplitude = nn.Parameter(torch.randn(1, self.k, max_len, input_dim) * 0.1)
        self.log_var_prior = nn.Parameter(torch.randn(1, max_len, input_dim) * 0.1 - 1)

    def load_pretrained(self, pretrained_model: "PULSESinusoid"):
        super().load_pretrained(pretrained_model)
        if self.k != pretrained_model.k:
            raise ValueError(
                f"k mismatch: {self.k} != {pretrained_model.k}. Please set k to the same value."
            )
        self.frequency = copy.deepcopy(pretrained_model.frequency)
        self.phase = copy.deepcopy(pretrained_model.phase)
        self.amplitude = copy.deepcopy(pretrained_model.amplitude)
        self.log_var_prior = copy.deepcopy(pretrained_model.log_var_prior)

    def forward(
        self,
        input_tensor: torch.Tensor,
        input_feature_mask: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        input_tensor: batch_size x seq_len x n_features
        input_feature_mask: batch_size x seq_len x n_features
        src_key_padding_mask: batch_size x seq_len

        Returns:
            mu_x: batch_size x seq_len x n_features
            var_x: batch_size x seq_len x n_features
            mu_p: batch_size x seq_len x n_features
            var_p: batch_size x seq_len x n_features
        """
        # Call parent PulseNormal forward method
        mu_x, var_x = super().forward(
            input_tensor=input_tensor,
            input_feature_mask=input_feature_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

        # Get the actual sequence length from the input
        seq_len = input_tensor.shape[1]
        batch_size = input_tensor.shape[0]

        # Compute sinusoidal prior: p(z) ~ N(A * sin(theta * pos + phase), sigma^2_p)
        # Parameters are already shaped as (1, k, max_len, input_dim)
        amplitude = self.amplitude[:, :, :seq_len, :].expand(
            batch_size, -1, -1, -1
        )  # (1, k, seq_len, input_dim)
        phase = self.phase[:, :, :seq_len, :]  # (1, k, seq_len, input_dim)
        frequency = self.frequency[:, :, :seq_len, :]  # (1, k, seq_len, input_dim)

        # pos is (1, 1, seq_len, 1)
        pos = self.positions[:, :, :seq_len, :]
        # Now broadcasting works directly: (batch_size, k, seq_len, input_dim)
        sines = amplitude * torch.sin(frequency * pos + phase)
        mu_p = torch.sum(
            sines, dim=1
        )  # sum over k dimension -> (batch_size, seq_len, input_dim)
        var_p = torch.exp(self.log_var_prior)[:, :seq_len, :].expand(batch_size, -1, -1)

        # Clamp var_p to prevent numerical instability
        var_p = torch.clamp(var_p, min=VAR_MIN, max=VAR_MAX)

        return mu_x, var_x, mu_p, var_p
