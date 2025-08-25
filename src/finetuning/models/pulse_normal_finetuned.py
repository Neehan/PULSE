"""
Fine-tuned PULSE Normal model for downstream tasks.
Inherits from AutoencoderFinetuned but uses PULSE Normal encoder with VAE sampling.
"""

import copy
from typing import Optional, Tuple

import torch

from src.finetuning.models.autoencoder_finetuned import AutoencoderFinetuned
from src.models.pulse_normal import PULSENormal


class PULSENormalFinetuned(AutoencoderFinetuned):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_pretrained(self, pretrained_model: PULSENormal):
        # override to restrict input model class
        super().load_pretrained(pretrained_model)

    def forward(
        self,
        input_tensor: torch.Tensor,
        input_feature_mask: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass that returns predictions, mu_x, var_x, z for VAE loss computation.
        """
        # Call parent's forward to get mu_x, var_x from PULSENormal encoder
        mu_x, var_x = super(AutoencoderFinetuned, self).forward(
            input_tensor, input_feature_mask, src_key_padding_mask
        )

        # Sample z using reparameterization trick
        eps = torch.randn_like(mu_x)
        z = mu_x + torch.sqrt(var_x) * eps

        # For prediction, use sampled values for masked dims, original for unmasked
        reconstructed = (z * input_feature_mask) + (
            input_tensor * (~input_feature_mask)
        )

        # Use parent's prediction helper
        predictions = self._predict_from_features(reconstructed)

        return predictions, mu_x, var_x
