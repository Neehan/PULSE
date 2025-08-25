from typing import Dict, Optional
import torch

from src.pretraining.trainers.autoencoder_trainer import AutoencoderTrainer
from src.models.pulse_normal import PULSENormal
from src.utils.constants import INPUT_DIM, OUTPUT_DIM
from src.utils.losses import compute_gaussian_kl_divergence, gaussian_log_likelihood


class PulseNormalTrainer(AutoencoderTrainer):
    """
    PulseNormal trainer that inherits from AutoencoderTrainer but implements VAE-style loss
    (reconstruction + KL divergence) for ICU vitals.
    """

    def __init__(
        self,
        model: PULSENormal,
        initial_n_masked_features: int,
        max_n_masked_features: int,
        n_masked_features_increase_every_n_epochs: int,
        alpha: float,
        **kwargs,
    ):
        super().__init__(
            model=model,
            initial_n_masked_features=initial_n_masked_features,
            max_n_masked_features=max_n_masked_features,
            n_masked_features_increase_every_n_epochs=n_masked_features_increase_every_n_epochs,
            **kwargs,
        )
        self.alpha = alpha

        # Update config to include alpha
        self.output_json["model_config"]["config"]["alpha"] = alpha

        # Override the losses collected for VAE training
        self.output_json["losses"] = {
            "train": {
                "total_loss": [],
                "reconstruction": [],
                "kl_term": [],
            },
            "val": {
                "total_loss": [],
                "reconstruction": [],
                "kl_term": [],
            },
        }

    def compute_kl_loss(
        self,
        input_tensor: torch.Tensor,
        input_feature_mask: torch.Tensor,
        mu_x: torch.Tensor,
        var_x: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence loss between posterior and standard normal prior."""
        # Standard normal prior for VAE: mu_p = 0, var_p = 1
        mu_p = torch.zeros_like(mu_x)
        var_p = torch.ones_like(var_x)

        kl_term = compute_gaussian_kl_divergence(
            input_feature_mask, mu_x, var_x, mu_p, var_p
        )
        return kl_term

    def compute_elbo_loss(
        self,
        input_tensor: torch.Tensor,
        input_feature_mask: torch.Tensor,
        mu_x: torch.Tensor,
        var_x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute ELBO loss = reconstruction loss + alpha * KL divergence.

        Args:
            input_tensor: Ground truth values
            input_feature_mask: Mask indicating which features are masked
            mu_x: Predicted mean values
            var_x: Predicted variance values
        """
        # Reconstruction term: negative log-likelihood of masked features
        n_masked_features = input_feature_mask.sum(dim=(1, 2)).float().mean()
        reconstruction_term = (
            -gaussian_log_likelihood(input_tensor, mu_x, var_x, input_feature_mask)
            / n_masked_features
        ).mean()

        # KL divergence term
        kl_term = (
            self.alpha
            * self.compute_kl_loss(input_tensor, input_feature_mask, mu_x, var_x).mean()
        ) / n_masked_features

        # Total ELBO loss
        total_loss = reconstruction_term + kl_term

        return {
            "total_loss": total_loss,
            "reconstruction": reconstruction_term,
            "kl_term": kl_term,
        }

    def compute_train_loss(
        self,
        input_tensor: torch.Tensor,
        input_feature_mask: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute PulseNormal training loss using VAE-style ELBO loss."""
        # Get model predictions - returns (mu_x, var_x)
        mu_x, var_x = self.model(input_tensor, input_feature_mask, src_key_padding_mask)

        # Compute ELBO loss
        loss_dict = self.compute_elbo_loss(
            input_tensor, input_feature_mask, mu_x, var_x
        )

        return loss_dict

    def compute_validation_loss(
        self,
        input_tensor: torch.Tensor,
        input_feature_mask: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute PulseNormal validation loss using VAE-style ELBO loss."""
        # Get model predictions - returns (mu_x, var_x)
        mu_x, var_x = self.model(input_tensor, input_feature_mask, src_key_padding_mask)

        # Compute ELBO loss
        loss_dict = self.compute_elbo_loss(
            input_tensor, input_feature_mask, mu_x, var_x
        )

        return loss_dict


def pulse_normal_training_loop(args_dict):
    """
    PulseNormal training loop using the PulseNormalTrainer class.
    Initializes the model internally and handles all training.
    """
    # Get distributed training parameters
    rank = args_dict.get("rank", 0)
    world_size = args_dict.get("world_size", 1)
    local_rank = args_dict.get("local_rank", 0)

    # Set device for this process
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Initialize PulseNormal model
    model = PULSENormal(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        device=str(device),
        **args_dict["model_size_params"],
    ).to(device)

    trainer = PulseNormalTrainer(
        model=model,
        input_dim=INPUT_DIM,
        initial_n_masked_features=args_dict["initial_n_masked_features"],
        max_n_masked_features=args_dict["max_n_masked_features"],
        n_masked_features_increase_every_n_epochs=args_dict[
            "n_masked_features_increase_every_n_epochs"
        ],
        alpha=args_dict["alpha"],
        batch_size=args_dict["batch_size"],
        num_epochs=args_dict["n_epochs"],
        init_lr=args_dict["init_lr"],
        scheduler_type=args_dict["scheduler_type"],
        training_type=args_dict["training_type"],
        num_warmup_epochs=args_dict["n_warmup_epochs"],
        decay_factor=args_dict["decay_factor"],
        pretrained_model_path=args_dict["pretrained_model_path"],
        resume_from_checkpoint=args_dict["resume_from_checkpoint"],
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        early_stopping_patience=args_dict["early_stopping_patience"],
        early_stopping_min_delta=args_dict["early_stopping_min_delta"],
    )

    return trainer.train(early_stopping=True)
