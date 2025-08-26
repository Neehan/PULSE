from typing import Optional
import torch.optim as optim
import os
import torch
import torch.distributed as dist
import logging
from argparse import ArgumentParser
import math
import random
import numpy as np
from .enums import SchedulerType


# --------------------------------------------
# Public Functions
# --------------------------------------------


def get_scheduler(
    optimizer: optim.Optimizer,
    num_warmup_epochs: int,
    total_epochs: int,
    decay_factor: Optional[float],
    scheduler_type: str,
):
    """
    Create a learning rate scheduler with warmup followed by cosine, exponential, or linear_flat annealing.

    Args:
        optimizer: PyTorch optimizer
        num_warmup_epochs: Number of epochs for linear warmup
        total_epochs: Total number of training epochs
        decay_factor: Decay factor for exponential annealing. Not used if scheduler_type is "cosine" or "linear_flat".
        scheduler_type: Type of scheduler to use, either "cosine", "exponential", or "linear_flat"
    Returns:
        PyTorch LR scheduler
    """

    if scheduler_type == SchedulerType.COSINE.value:
        # Use cosine annealing
        lr_lambda = _cosine_annealing_lr(num_warmup_epochs, total_epochs)
    elif scheduler_type == SchedulerType.EXPONENTIAL.value:
        if decay_factor is None:
            raise ValueError("Decay factor is required for exponential scheduler")
        # Use exponential annealing
        lr_lambda = _exponential_annealing_lr(
            num_warmup_epochs, total_epochs, decay_factor
        )
    elif scheduler_type == SchedulerType.LINEAR_FLAT.value:
        # Use linear warmup then flat
        lr_lambda = _linear_flat_lr(num_warmup_epochs)
    else:
        valid_types = SchedulerType.get_all_values()
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type}. Valid types: {valid_types}"
        )

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def setup_distributed():
    """Initialize distributed training environment"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        # Initialize the process group
        dist.init_process_group(backend="nccl")

        # Set device for this process
        torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank
    else:
        # Single GPU training
        return 0, 1, 0


def cleanup_distributed():
    """Clean up distributed training environment"""
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Configure logging only for rank 0
def setup_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
    else:
        logging.basicConfig(level=logging.WARNING)


def get_model_params(model_size: str):
    if model_size == "mini":
        model_size_params = {"num_heads": 4, "num_layers": 2, "head_dim": 12}
    elif model_size == "small":
        model_size_params = {"num_heads": 8, "num_layers": 4, "head_dim": 24}
    elif model_size == "medium":
        model_size_params = {"num_heads": 12, "num_layers": 6, "head_dim": 36}
    elif model_size == "large":
        model_size_params = {"num_heads": 16, "num_layers": 8, "head_dim": 48}
    else:
        raise ValueError(f"Unknown model size: {model_size}")
    return model_size_params


def parse_args(parser: ArgumentParser) -> dict:
    args = parser.parse_args()
    args_dict = vars(args)

    logger = logging.getLogger(__name__)
    logger.info("Command-line arguments:")
    for arg, value in args_dict.items():
        logger.info(f"{arg}: {value}")

    # Model size configuration
    model_size = args.model_size.lower()
    model_size_params = get_model_params(model_size)

    args_dict["model_size_params"] = model_size_params

    return args_dict


# --------------------------------------------
# Helper Functions
# --------------------------------------------


def _cosine_annealing_lr(num_warmup_epochs, total_epochs):
    def lr_function(current_epoch):
        if current_epoch < num_warmup_epochs:
            # Linear warmup
            return float(current_epoch) / float(max(1, num_warmup_epochs))
        else:
            # Cosine annealing after warmup
            progress = (current_epoch - num_warmup_epochs) / (
                total_epochs - num_warmup_epochs
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_function


def _exponential_annealing_lr(num_warmup_epochs, total_epochs, decay_factor):
    def lr_function(current_epoch):
        if current_epoch < num_warmup_epochs:
            # Linear warmup
            return float(current_epoch) / float(max(1, num_warmup_epochs))
        else:
            # Exponential annealing after warmup
            epochs_after_warmup = current_epoch - num_warmup_epochs
            return decay_factor**epochs_after_warmup

    return lr_function


def _linear_flat_lr(num_warmup_epochs):
    def lr_function(current_epoch):
        if current_epoch < num_warmup_epochs:
            # Linear warmup
            return float(current_epoch) / float(max(1, num_warmup_epochs))
        else:
            # Flat learning rate after warmup
            return 1.0

    return lr_function
