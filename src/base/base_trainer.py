import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from src.base.base_model import BaseModel
from src.utils import utils
from src.utils.constants import DATA_DIR, DRY_RUN


class BaseTrainer(ABC):
    """
    Base trainer class that provides common training infrastructure.
    Args:
        model: BaseModel instance
        batch_size: Batch size
        num_epochs: Number of epochs to train
        init_lr: Initial learning rate
        scheduler_type: Type of scheduler to use (cosine or exponential)
        training_type: Type of training to be done (e.g. "pretraining", "finetuning")
        num_warmup_epochs: Number of epochs for linear warmup
        decay_factor: Decay factor for exponential scheduler. Not used if scheduler_type is "cosine".
        pretrained_model_path: Path to pretrained model
        resume_from_checkpoint: Path to checkpoint to resume from
        For distributed training:
            rank: Rank of the process (0 for main process, 1 for other processes)
            world_size: World size (number of processes)
            local_rank: Local rank of the process (0 for main process, 1 for other processes)

    PUBLIC METHODS:
        - train(): Main entry point to train the model
        - save_checkpoint(): Save training state
        - load_checkpoint(): Resume from checkpoint

    ABSTRACT METHODS (must be implemented by children):
        - get_dataloaders(): Get data loaders for training/validation
        - compute_train_loss(): Compute training loss for a batch
        - compute_validation_loss(): Compute validation loss for a batch
        - get_model_name(): Get model name for saving
    """

    def __init__(
        self,
        model: BaseModel,
        input_dim: int,
        batch_size: int,
        num_epochs: int,
        init_lr: float,
        scheduler_type: str,
        training_type: str,
        num_warmup_epochs: int = 10,
        decay_factor: Optional[float] = 0.99,
        pretrained_model_path: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None,
        rank: int = 0,
        world_size: int = 1,
        local_rank: int = 0,
        # Early stopping
        early_stopping_patience: int = 15,
        early_stopping_min_delta: float = 1e-3,
    ):
        self.model: Union[BaseModel, DDP]  # Add type annotation for model attribute

        self._setup_distributed_training(rank, world_size, local_rank)
        # Setup basic logging early so logger is available in other methods
        utils.setup_logging(self.rank)
        self.logger = logging.getLogger(__name__)

        self._setup_model_and_device(model, input_dim, batch_size, num_epochs)
        self._setup_model_directory(training_type)
        self._load_pretrained_model(pretrained_model_path)
        self._setup_training_components(
            init_lr, num_warmup_epochs, decay_factor, scheduler_type
        )
        self._setup_logging_and_output()
        self._setup_early_stopping_params(
            early_stopping_patience, early_stopping_min_delta
        )
        self._resume_from_checkpoint(resume_from_checkpoint)

    # =============================================================================
    # PUBLIC API METHODS (for users)
    # =============================================================================

    def get_model_name(self) -> str:
        """Get the model name for saving - MUST BE IMPLEMENTED BY CHILDREN."""
        model = self._get_underlying_model()
        name = f"{model.name}_{model.total_params_formatted()}"
        return name

    def get_current_epoch(self) -> Optional[int]:
        """Get the current epoch - PUBLIC API METHOD."""
        return self.current_epoch

    def get_num_epochs(self) -> int:
        """Get the number of epochs to train - PUBLIC API METHOD."""
        return self.num_epochs

    def train(self, early_stopping: bool = True) -> float:
        """
        Main training loop - PUBLIC API METHOD.

        Args:
            early_stopping: Whether to use early stopping. When enabled, the best model
                           (based on validation loss) is automatically saved as
                           "{model_name}_best.pth" and restored if early stopping is triggered.

        Returns:
            Best validation loss achieved during training
        """
        for epoch in range(self.start_epoch, self.num_epochs):
            train_loader, val_loader = self.get_dataloaders(shuffle=True)

            self.current_epoch = epoch

            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate_epoch(val_loader)

            # Early stopping check and best model saving
            if early_stopping and self._check_early_stopping(val_loss):
                if self.rank == 0:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    # Restore best weights from saved best model
                    if self.early_stopping_best_epoch is not None:
                        self._restore_best_model_for_early_stopping()
                break

            if self.rank == 0:
                self.logger.info(
                    f"Epoch [{epoch+1} / {self.num_epochs}]: Train loss: {train_loss:.3f} Validation loss: {val_loss:.3f}"
                )

                if epoch % 5 == 1 or epoch == self.num_epochs - 1:
                    self.save_checkpoint(epoch)

                self._save_output_json()

        # Clean up numbered checkpoints after training is complete
        if self.rank == 0:
            self._cleanup_numbered_checkpoints()

        return self.early_stopping_best_score

    def save_checkpoint(self, epoch: int):
        """Save a complete checkpoint - PUBLIC API METHOD."""
        if self.rank != 0:
            return

        model_to_save = self._get_underlying_model()
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.early_stopping_best_score,
            "output_json": self.output_json,
        }

        model_name = self.get_model_name()
        # Save epoch-specific and latest checkpoints
        checkpoint_path = self.model_dir + f"{model_name}_epoch_{epoch}_checkpoint.pth"
        latest_checkpoint_path = self.model_dir + f"{model_name}_latest_checkpoint.pth"
        model_path = self.model_dir + f"{model_name}_epoch_{epoch}.pth"
        latest_model_path = self.model_dir + f"{model_name}_latest.pth"

        torch.save(checkpoint, checkpoint_path)
        torch.save(checkpoint, latest_checkpoint_path)
        torch.save(model_to_save, model_path)
        torch.save(model_to_save, latest_model_path)

        # Track the numbered files for cleanup
        self.saved_checkpoint_files.extend([checkpoint_path, model_path])

    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint to resume training - PUBLIC API METHOD."""
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        model_to_load = self._get_underlying_model()
        model_to_load.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.start_epoch = checkpoint["epoch"]

        # Load best validation loss and output json
        self.early_stopping_best_score = checkpoint["best_val_loss"]
        self.output_json = checkpoint["output_json"]

        if self.rank == 0:
            self.logger.info(
                f"Loaded checkpoint from {checkpoint_path}, resuming from epoch {self.start_epoch}"
            )

    # =============================================================================
    # ABSTRACT METHODS (must be implemented by children)
    # =============================================================================

    @abstractmethod
    def get_dataloaders(
        self, shuffle: bool = True, cross_validation_k: Optional[int] = None
    ) -> Tuple[DataLoader, DataLoader]:
        """Get data loaders for training/validation - MUST BE IMPLEMENTED BY CHILDREN.

        Args:
            shuffle: Whether to shuffle the data
            cross_validation_k: If provided, return the k-th fold of train/val loaders

        Returns:
            Tuple of (train_loader, val_loader)
        """
        raise NotImplementedError("get_dataloaders must be implemented by children")

    @abstractmethod
    def compute_train_loss(
        self, *input_data: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss for a batch - MUST BE IMPLEMENTED BY CHILDREN.

        Args:
            *input_data: Input tensors for the batch

        Returns:
            Dict containing 'total_loss' and other loss components
        """
        raise NotImplementedError("compute_train_loss must be implemented by children")

    @abstractmethod
    def compute_validation_loss(
        self, *input_data: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute validation loss for a batch - MUST BE IMPLEMENTED BY CHILDREN.

        Args:
            *input_data: Input tensors for the batch

        Returns:
            Dict containing 'total_loss' and other loss components
        """
        raise NotImplementedError(
            "compute_validation_loss must be implemented by children"
        )

    # =============================================================================
    # INTERNAL TRAINING METHODS
    # =============================================================================

    def _masked_mean(
        self, tensor: torch.Tensor, mask: torch.Tensor, dim: Tuple[int, ...]
    ):
        """Mean over `dim`, ignoring False in `mask`."""
        masked = tensor * mask
        return masked.sum(dim=dim) / (mask.sum(dim=dim).clamp(min=1))

    def _train_epoch(self, loader) -> float:
        """Train the model for one epoch."""
        self.model.train()
        total_loss_dict = self._initialize_loss_dict("train")

        if self.rank == 0:
            self.logger.info("Started training epoch.")

        loader_len = 0
        for input_data in loader:
            input_data = [
                input_i.to(self.device) if input_i is not None else None
                for input_i in input_data
            ]
            self.optimizer.zero_grad()

            loss_dict = self.compute_train_loss(*input_data)
            loss = loss_dict["total_loss"]
            if self.rank == 0 and DRY_RUN:
                print(f"Train loss: {loss.item()}")

            self._accumulate_losses(total_loss_dict, loss_dict)
            loader_len += 1

            loss.backward()
            self.optimizer.step()

        self.scheduler.step()
        self._sync_distributed_training()

        avg_loss_dict = self._average_losses(total_loss_dict, loader_len)
        self._update_output_json_losses("train", avg_loss_dict)

        return avg_loss_dict["total_loss"]

    def _validate_epoch(self, loader) -> float:
        """Validate the model for one epoch."""
        self.model.eval()
        total_loss_dict = self._initialize_loss_dict("val")

        if self.rank == 0:
            self.logger.info("Started validation epoch.")

        loader_len = 0
        for input_data in loader:
            input_data = [
                input_i.to(self.device) if input_i is not None else None
                for input_i in input_data
            ]

            with torch.no_grad():
                loss_dict = self.compute_validation_loss(*input_data)

            self._accumulate_losses(total_loss_dict, loss_dict)
            loader_len += 1

        self._sync_distributed_training()

        avg_loss_dict = self._average_losses(total_loss_dict, loader_len)
        self._update_output_json_losses("val", avg_loss_dict)

        return avg_loss_dict["total_loss"]

    # =============================================================================
    # HELPER/UTILITY METHODS
    # =============================================================================

    def _check_early_stopping(self, current_score: float) -> bool:
        """Simple early stopping check. Returns True if should stop."""
        # Check if score improved by min_delta
        if (
            current_score
            < self.early_stopping_best_score - self.early_stopping_min_delta
        ):
            self.early_stopping_best_score = current_score
            self.early_stopping_best_epoch = self.current_epoch
            self.early_stopping_counter = 0
            # Save the best model when we find a better score
            self._save_best_model()
        else:
            self.early_stopping_counter += 1

        return self.early_stopping_counter >= self.early_stopping_patience

    def _setup_early_stopping_params(self, patience: int, min_delta: float):
        """Setup early stopping parameters."""
        self.early_stopping_patience = patience
        self.early_stopping_min_delta = min_delta
        self.early_stopping_counter = 0
        self.early_stopping_best_score = float("inf")
        self.early_stopping_best_epoch = None

    def _setup_distributed_training(self, rank: int, world_size: int, local_rank: int):
        """Setup distributed training parameters."""
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.is_distributed = world_size > 1

    def _setup_model_and_device(
        self, model: BaseModel, input_dim: int, batch_size: int, num_epochs: int
    ):
        """Setup model, device, and batch size."""
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.current_epoch = None
        self.device = torch.device(
            f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        )

        self.model = model.to(self.device)

        if self.is_distributed:
            self.model = DDP(
                self.model, device_ids=[self.local_rank], output_device=self.local_rank
            )
            self.batch_size = batch_size // self.world_size

    def _setup_training_components(
        self,
        init_lr: float,
        num_warmup_epochs: int,
        decay_factor: Optional[float],
        scheduler_type: str,
    ):
        """
        Setup optimizer and scheduler.

        CRITICAL: This must be called AFTER _load_pretrained_model() because:
        - The optimizer captures parameter references at creation time
        - If we load pretrained weights after optimizer creation, the optimizer
          will still reference the old (random) parameters, not the new pretrained ones
        - This was a major bug that prevented pretrained models from being optimized!
        """
        self.init_lr = init_lr
        self.num_warmup_epochs = num_warmup_epochs
        self.decay_factor = decay_factor

        # Create optimizer with current model parameters (which should be pretrained if provided)
        self.optimizer = optim.Adam(self.model.parameters(), lr=init_lr)
        if self.rank == 0:
            if decay_factor is None:
                self.logger.info("using cosine annealing")
            else:
                self.logger.info(
                    f"using exponential annealing with decay factor {decay_factor}"
                )

        self.scheduler = utils.get_scheduler(
            self.optimizer,
            num_warmup_epochs,
            self.num_epochs,
            decay_factor,
            scheduler_type,
        )

    def _setup_logging_and_output(self):
        """Setup logging and output JSON structure."""
        if self.rank == 0:
            self.logger.info(str(self._get_underlying_model()))
            model_for_params = self._get_underlying_model()
            self.logger.info(
                f"Total number of parameters: {model_for_params.total_params_formatted()}"
            )
            self.logger.info(
                f"Distributed training: {self.is_distributed}, World size: {self.world_size}"
            )
            self.logger.info(f"Batch size per GPU: {self.batch_size}")

        model_for_config = self._get_underlying_model()
        self.output_json = {
            "model_config": {
                "total_params": model_for_config.total_params(),
                "batch_size": self.batch_size * self.world_size,  # Original batch size
                "batch_size_per_gpu": self.batch_size,
                "world_size": self.world_size,
                "init_lr": self.init_lr,
                "num_warmup_epochs": self.num_warmup_epochs,
                "decay_factor": self.decay_factor,
                "model_layers": str(model_for_config),
            },
            "losses": {"train": {"total_loss": []}, "val": {"total_loss": []}},
        }

        # Track saved checkpoint files for cleanup
        self.saved_checkpoint_files: List[str] = []

    def _setup_model_directory(self, training_type: str):
        """Setup model directory for saving.
        Args:
            training_type: Type of training to be done (e.g. "pretraining", "finetuning")
        """
        if self.rank == 0:
            self.model_dir = DATA_DIR + f"trained_models/{training_type}/"
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)

    def _load_pretrained_model(self, pretrained_model_path: Optional[str]):
        """Load pretrained model if provided."""
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            pretrained_model = torch.load(pretrained_model_path, weights_only=False)
            self.logger.info(
                f"🔄 Loading pretrained model from: {pretrained_model_path}"
            )
            self._get_underlying_model().load_pretrained(pretrained_model)
            self.logger.info(
                "✅ Pretrained model loaded successfully - optimizer will use these weights"
            )

    def _resume_from_checkpoint(self, resume_from_checkpoint: Optional[str]):
        """Resume from checkpoint if provided."""
        self.start_epoch = 0
        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            self.load_checkpoint(resume_from_checkpoint)

    def _get_underlying_model(self) -> BaseModel:
        """Get the underlying model, unwrapping DDP if necessary."""
        if self.is_distributed and isinstance(self.model, DDP):
            return self.model.module  # type: ignore
        else:
            return self.model  # type: ignore

    def _initialize_loss_dict(self, split: str) -> Dict[str, float]:
        """Initialize loss dictionary for accumulation."""
        return {key: 0.0 for key in self.output_json["losses"][split]}

    def _accumulate_losses(
        self, total_loss_dict: Dict[str, float], loss_dict: Dict[str, torch.Tensor]
    ):
        """Accumulate losses from a batch."""
        for key in loss_dict:
            total_loss_dict[key] += loss_dict[key].item()

    def _sync_distributed_training(self):
        """Synchronize all processes in distributed training."""
        if self.is_distributed:
            dist.barrier()

    def _average_losses(
        self, total_loss_dict: Dict[str, float], loader_len: int
    ) -> Dict[str, float]:
        """Average losses across batches and processes."""
        avg_loss_dict = {}

        for key in total_loss_dict:
            avg_loss_dict[key] = total_loss_dict[key] / loader_len

            if self.is_distributed:
                loss_tensor = torch.tensor(avg_loss_dict[key], device=self.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                avg_loss_dict[key] = loss_tensor.item() / self.world_size

        return avg_loss_dict

    def _update_output_json_losses(self, split: str, avg_loss_dict: Dict[str, float]):
        """Update output JSON with averaged losses."""
        if self.rank == 0:
            for key in self.output_json["losses"][split]:
                self.output_json["losses"][split][key].append(avg_loss_dict[key])

    def _save_output_json(self):
        """Save the output JSON containing model config and losses."""
        if self.rank != 0:
            return

        model_name = self.get_model_name()
        filename = f"{model_name}_output.json"
        with open(self.model_dir + filename, "w") as f:
            json.dump(self.output_json, f, indent=2)

    def _cleanup_numbered_checkpoints(self):
        """Remove all numbered checkpoint and model files, keeping only the latest versions."""
        if self.rank != 0:
            return

        files_to_remove = self.saved_checkpoint_files

        if files_to_remove:
            self.logger.info(
                f"Cleaning up {len(files_to_remove)} numbered checkpoint and model files..."
            )
            for file_path in files_to_remove:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except OSError as e:
                    self.logger.warning(
                        f"Failed to remove {os.path.basename(file_path)}: {e}"
                    )
                    # Continue with other files instead of failing the entire training

            self.logger.info(
                "Cleanup completed. Only latest checkpoint and model files remain."
            )
            self.saved_checkpoint_files.clear()

    def _save_best_model(self):
        """Save the model with the best validation loss."""
        if self.rank != 0:
            return

        model_to_save = self._get_underlying_model()
        model_name = self.get_model_name()
        best_model_path = self.model_dir + f"{model_name}_best.pth"

        torch.save(model_to_save, best_model_path)
        self.logger.info(
            f"Saved best model with validation loss {self.early_stopping_best_score:.4f} to {best_model_path}"
        )

    def _restore_best_model_for_early_stopping(self):
        """Restore the best model when early stopping is triggered."""
        if self.rank != 0:
            return

        model_name = self.get_model_name()
        best_model_path = self.model_dir + f"{model_name}_best.pth"

        if os.path.exists(best_model_path):
            # Load the best model
            best_model = torch.load(
                best_model_path, map_location=self.device, weights_only=False
            )
            model_to_load = self._get_underlying_model()
            model_to_load.load_state_dict(best_model.state_dict())
            self.logger.info(
                f"Restored best model weights for early stopping (from epoch {self.early_stopping_best_epoch})"
            )
        else:
            self.logger.warning(f"Best model file not found, keeping current weights")
