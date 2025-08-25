from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.base.base_trainer import BaseTrainer
from src.finetuning.models.autoencoder_finetuned import AutoencoderFinetuned
from src.finetuning.dataloaders.finetuning_dataloader import FinetunedDataloader
from src.utils.constants import (
    INPUT_DIM,
    DEFAULT_RANK,
    DEFAULT_WORLD_SIZE,
    DEFAULT_LOCAL_RANK,
    SQUEEZE_LAST_DIM,
    TRAIN_SPLIT,
    VAL_SPLIT,
)


class AutoencoderFinetunedTrainer(BaseTrainer):
    """
    Fine-tuning trainer for AutoencoderFinetuned model.
    Handles supervised learning tasks with input sequences and target labels.
    """

    def __init__(
        self,
        model: AutoencoderFinetuned,
        task_type: str,  # "regression" or "classification"
        prediction_dim: int,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.task_type = task_type
        self.prediction_dim = prediction_dim

        # Set up loss function based on task type
        if task_type == "regression":
            self.criterion = nn.MSELoss()
        elif task_type == "classification":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(
                f"Unsupported task_type: {task_type}. Use 'regression' or 'classification'"
            )

        # Update output json with task info
        self.output_json["model_config"]["task_type"] = task_type
        self.output_json["model_config"]["prediction_dim"] = prediction_dim
        self.output_json["model_config"]["criterion"] = str(self.criterion)

    def compute_train_loss(
        self,
        input_tensor: torch.Tensor,
        input_feature_mask: torch.Tensor,
        target: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss for fine-tuning.

        Args:
            input_tensor: batch_size x seq_len x n_features
            input_feature_mask: batch_size x seq_len x n_features
            target: batch_size x prediction_dim (regression) or batch_size (classification)
            src_key_padding_mask: Optional padding mask (usually not needed for fine-tuning)
        """
        predictions = self.model(input_tensor, input_feature_mask, src_key_padding_mask)

        if self.task_type == "classification" and target.dim() > 1:
            target = target.squeeze(-1)  # Remove extra dimension if present

        loss = self.criterion(predictions, target)
        return {"total_loss": loss}

    def compute_validation_loss(
        self,
        input_tensor: torch.Tensor,
        input_feature_mask: torch.Tensor,
        target: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute validation loss for fine-tuning.
        """
        predictions = self.model(input_tensor, input_feature_mask, src_key_padding_mask)

        if self.task_type == "classification" and target.dim() > 1:
            target = target.squeeze(-1)

        loss = self.criterion(predictions, target)
        return {"total_loss": loss}

    def get_dataloaders(self, shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
        """
        Get data loaders for fine-tuning using synthetic data.
        """
        train_dataloader = FinetunedDataloader(
            batch_size=self.batch_size,
            split=TRAIN_SPLIT,
            shuffle=shuffle,
            task_type=self.task_type,
            prediction_dim=self.prediction_dim,
            world_size=self.world_size,
            rank=self.rank,
        )

        val_dataloader = FinetunedDataloader(
            batch_size=self.batch_size,
            split=VAL_SPLIT,
            shuffle=False,  # Don't shuffle validation data
            task_type=self.task_type,
            prediction_dim=self.prediction_dim,
            world_size=self.world_size,
            rank=self.rank,
        )

        return train_dataloader, val_dataloader


def autoencoder_finetuned_training_loop(args_dict):
    """
    Fine-tuning training loop for AutoencoderFinetuned model.
    """
    # Get distributed training parameters
    rank = args_dict.get("rank", DEFAULT_RANK)
    world_size = args_dict.get("world_size", DEFAULT_WORLD_SIZE)
    local_rank = args_dict.get("local_rank", DEFAULT_LOCAL_RANK)

    # Set device for this process
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Initialize fine-tuned model
    model = AutoencoderFinetuned(
        input_dim=INPUT_DIM,
        output_dim=INPUT_DIM,  # This gets overridden by parent class anyway
        device=str(device),
        prediction_dim=args_dict["prediction_dim"],
        attention_hidden_dim=args_dict["attention_hidden_dim"],
        mlp_hidden_dim=args_dict["mlp_hidden_dim"],
        dropout_rate=args_dict["dropout_rate"],
        **args_dict["model_size_params"],
    ).to(device)

    trainer = AutoencoderFinetunedTrainer(
        model=model,
        input_dim=INPUT_DIM,
        task_type=args_dict["task_type"],
        prediction_dim=args_dict["prediction_dim"],
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
