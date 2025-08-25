from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.finetuning.trainers.autoencoder_finetuned_trainer import (
    AutoencoderFinetunedTrainer,
)
from src.finetuning.models.pulse_normal_finetuned import PULSENormalFinetuned
from src.finetuning.dataloaders.finetuning_dataloader import FinetunedDataloader
from src.utils.losses import gaussian_log_likelihood, compute_gaussian_kl_divergence
from src.utils.constants import (
    INPUT_DIM,
    DEFAULT_RANK,
    DEFAULT_WORLD_SIZE,
    DEFAULT_LOCAL_RANK,
    TRAIN_SPLIT,
    VAL_SPLIT,
)


class PULSENormalFinetunedTrainer(AutoencoderFinetunedTrainer):
    """
    Fine-tuning trainer for PULSENormalFinetuned model with VAE loss and beta parameter.
    """

    def __init__(
        self,
        model: PULSENormalFinetuned,
        beta: float,  # Beta parameter for KL weighting
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        self.beta = beta

        # Update output json with beta parameter
        self.output_json["model_config"]["beta"] = beta

        # Override the losses collected for VAE training
        self.output_json["losses"] = {
            "train": {
                "total_loss": [],
                "task_loss": [],
                "reconstruction": [],
                "kl_term": [],
            },
            "val": {
                "total_loss": [],
                "task_loss": [],
                "reconstruction": [],
                "kl_term": [],
            },
        }

    def compute_train_loss(
        self,
        input_tensor: torch.Tensor,
        input_feature_mask: torch.Tensor,
        target: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss with VAE reconstruction + KL + task loss.
        """
        predictions, mu_x, var_x = self.model(
            input_tensor, input_feature_mask, src_key_padding_mask
        )

        # Task loss (classification/regression)
        if self.task_type == "classification" and target.dim() > 1:
            target = target.squeeze(-1)
        task_loss = self.criterion(predictions, target)

        # VAE losses
        n_masked_features = input_feature_mask.sum(dim=(1, 2)).float().mean()

        # Reconstruction loss: negative log-likelihood of masked features
        reconstruction_loss = (
            -gaussian_log_likelihood(input_tensor, mu_x, var_x, input_feature_mask)
            / n_masked_features
        ).mean()

        # KL divergence loss (standard normal prior)
        mu_p = torch.zeros_like(mu_x)
        var_p = torch.ones_like(var_x)
        kl_loss = (
            compute_gaussian_kl_divergence(
                input_feature_mask, mu_x, var_x, mu_p, var_p
            ).mean()
            / n_masked_features
        )

        # Total loss
        total_loss = task_loss + reconstruction_loss + self.beta * kl_loss

        return {
            "total_loss": total_loss,
            "task_loss": task_loss,
            "reconstruction": reconstruction_loss,
            "kl_term": kl_loss,
        }

    def compute_validation_loss(
        self,
        input_tensor: torch.Tensor,
        input_feature_mask: torch.Tensor,
        target: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute validation loss with VAE reconstruction + KL + task loss.
        """
        predictions, mu_x, var_x, z = self.model(
            input_tensor, input_feature_mask, src_key_padding_mask
        )

        # Task loss
        if self.task_type == "classification" and target.dim() > 1:
            target = target.squeeze(-1)
        task_loss = self.criterion(predictions, target)

        # VAE losses
        n_masked_features = input_feature_mask.sum(dim=(1, 2)).float().mean()

        reconstruction_loss = (
            -gaussian_log_likelihood(input_tensor, mu_x, var_x, input_feature_mask)
            / n_masked_features
        ).mean()

        mu_p = torch.zeros_like(mu_x)
        var_p = torch.ones_like(var_x)
        kl_loss = (
            compute_gaussian_kl_divergence(
                input_feature_mask, mu_x, var_x, mu_p, var_p
            ).mean()
            / n_masked_features
        )

        total_loss = task_loss + reconstruction_loss + self.beta * kl_loss

        return {
            "total_loss": total_loss,
            "task_loss": task_loss,
            "reconstruction": reconstruction_loss,
            "kl_term": kl_loss,
        }


def pulse_normal_finetuned_training_loop(args_dict):
    """
    Fine-tuning training loop for PULSENormalFinetuned model.
    """
    # Get distributed training parameters
    rank = args_dict.get("rank", DEFAULT_RANK)
    world_size = args_dict.get("world_size", DEFAULT_WORLD_SIZE)
    local_rank = args_dict.get("local_rank", DEFAULT_LOCAL_RANK)

    # Set device for this process
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Initialize fine-tuned model
    model = PULSENormalFinetuned(
        input_dim=INPUT_DIM,
        output_dim=INPUT_DIM,
        device=str(device),
        prediction_dim=args_dict["prediction_dim"],
        attention_hidden_dim=args_dict["attention_hidden_dim"],
        mlp_hidden_dim=args_dict["mlp_hidden_dim"],
        **args_dict["model_size_params"],
    ).to(device)

    trainer = PULSENormalFinetunedTrainer(
        model=model,
        input_dim=INPUT_DIM,
        task_type=args_dict["task_type"],
        prediction_dim=args_dict["prediction_dim"],
        beta=args_dict["beta"],
        batch_size=args_dict["batch_size"],
        num_epochs=args_dict["n_epochs"],
        init_lr=args_dict["init_lr"],
        scheduler_type=args_dict["scheduler_type"],
        training_type="finetuning",
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
