from typing import Dict, Optional
import torch
import torch.nn as nn

from src.base.base_trainer import BaseTrainer
from src.models.autoencoder import Autoencoder
from src.utils.constants import INPUT_DIM, OUTPUT_DIM
from src.pretraining.dataloaders.pretraining_dataloader import PretrainingDataloader
from src.utils.pretraining_masks import full_feature_mask_random
from torch.utils.data import DataLoader
from typing import Tuple


class AutoencoderTrainer(BaseTrainer):
    """
    Autoencoder trainer that implements masked time series prediction for ICU vitals.
    """

    def __init__(
        self,
        model: Autoencoder,
        initial_n_masked_features: int,
        max_n_masked_features: int,
        n_masked_features_increase_every_n_epochs: int,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.criterion = nn.MSELoss(reduction="mean")
        self.masking_function = full_feature_mask_random
        self.initial_n_masked_features = initial_n_masked_features
        self.max_n_masked_features = max_n_masked_features
        self.n_masked_features_increase_every_n_epochs = (
            n_masked_features_increase_every_n_epochs
        )

        self.output_json["model_config"][
            "masking_function"
        ] = "full_feature_mask_random"
        self.output_json["model_config"]["config"] = {
            "initial_n_masked_features": initial_n_masked_features,
            "max_n_masked_features": max_n_masked_features,
            "n_masked_features_increase_every_n_epochs": (
                n_masked_features_increase_every_n_epochs
            ),
        }

    def compute_train_loss(
        self,
        input_tensor: torch.Tensor,
        input_feature_mask: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute BERT training loss using MSE between predicted and actual masked tokens."""
        target_tokens = input_tensor[input_feature_mask]
        output = self.model(input_tensor, input_feature_mask, src_key_padding_mask)
        predicted_tokens = output[input_feature_mask]
        loss = self.criterion(target_tokens, predicted_tokens)

        return {"total_loss": loss}

    def compute_validation_loss(
        self,
        input_tensor: torch.Tensor,
        input_feature_mask: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute BERT validation loss using MSE between predicted and actual masked tokens."""
        target_tokens = input_tensor[input_feature_mask]

        output = self.model(input_tensor, input_feature_mask, src_key_padding_mask)
        predicted_tokens = output[input_feature_mask]
        loss = self.criterion(target_tokens, predicted_tokens)

        return {"total_loss": loss}

    def get_dataloaders(self, shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
        """Get data loaders for training/validation."""

        current_n_masked = self._get_current_n_masked_features(
            self.current_epoch,
        )

        train_loader = PretrainingDataloader(
            self.batch_size,
            split="train",
            shuffle=shuffle,
            masking_function=self.masking_function,
            n_masked_features=current_n_masked,
            world_size=self.world_size,
            rank=self.rank,
        )

        val_loader = PretrainingDataloader(
            self.batch_size,
            split="validation",
            shuffle=False,
            masking_function=self.masking_function,
            n_masked_features=current_n_masked,
            world_size=self.world_size,
            rank=self.rank,
        )

        return train_loader, val_loader

    def _get_current_n_masked_features(
        self,
        epoch: Optional[int],
    ):
        """Calculate dynamic n_masked_features: start with initial, increase by 2 every 5 epochs, cap at 20."""
        if epoch is None:
            return self.initial_n_masked_features
        # mask 1 more features every k epochs up to a max
        additional_features = (
            epoch // self.n_masked_features_increase_every_n_epochs
        ) * 1
        return min(
            self.initial_n_masked_features + additional_features,
            self.max_n_masked_features,
        )


def autoencoder_training_loop(args_dict):
    """
    Autoencoder training loop using the AutoencoderTrainer class.
    Initializes the model internally and handles all training.
    """
    # Get distributed training parameters
    rank = args_dict.get("rank", 0)
    world_size = args_dict.get("world_size", 1)
    local_rank = args_dict.get("local_rank", 0)

    # Set device for this process
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Initialize autoencoder model
    model = Autoencoder(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        device=str(device),
        **args_dict["model_size_params"],
    ).to(device)

    trainer = AutoencoderTrainer(
        model=model,
        input_dim=INPUT_DIM,
        initial_n_masked_features=args_dict["initial_n_masked_features"],
        max_n_masked_features=args_dict["max_n_masked_features"],
        n_masked_features_increase_every_n_epochs=args_dict[
            "n_masked_features_increase_every_n_epochs"
        ],
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
