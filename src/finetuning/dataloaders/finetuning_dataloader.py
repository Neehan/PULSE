import torch
import numpy as np
from torch.utils.data import DataLoader, IterableDataset
from src.utils.constants import (
    INPUT_DIM,
    MAX_CONTEXT_LENGTH,
    DRY_RUN,
    DRY_RUN_ITERATIONS,
    SYNTHETIC_BASELINE_MIN,
    SYNTHETIC_BASELINE_RANGE,
    SYNTHETIC_TREND_STD,
    SYNTHETIC_NOISE_STD,
    SYNTHETIC_PERIODIC_AMPLITUDE,
    SYNTHETIC_PERIODIC_CYCLES,
    SYNTHETIC_TARGET_PORTION,
    SYNTHETIC_REGRESSION_NOISE_STD,
    SYNTHETIC_CLASSIFICATION_THRESHOLD,
)
from typing import Tuple


class SyntheticFinetunedDataset(IterableDataset):
    """Generates synthetic ICU vitals with target labels for fine-tuning on-the-fly"""

    def __init__(
        self,
        split: str,
        rank: int,
        task_type: str = "regression",
        prediction_dim: int = 1,
    ):
        self.task_type = task_type
        self.prediction_dim = prediction_dim
        self.split = split
        self.device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        self.seq_len = min(512, MAX_CONTEXT_LENGTH)
        self.input_dim = INPUT_DIM

        # Set different random seed for different splits to ensure different data
        self.split_seed = hash(split) % (2**31)

    def __iter__(self):
        """Generate infinite stream of synthetic samples with targets"""
        # Set numpy random state based on split to ensure different data for train/val
        rng = np.random.RandomState(self.split_seed)

        while True:
            # Generate one sample of synthetic ICU vitals (same as pretraining)
            sample = torch.zeros(self.seq_len, self.input_dim)

            for feat_idx in range(self.input_dim):
                # Simple realistic time series: baseline + trend + noise + periodic
                baseline = (
                    SYNTHETIC_BASELINE_MIN + SYNTHETIC_BASELINE_RANGE * rng.random()
                )
                trend = rng.normal(0, SYNTHETIC_TREND_STD, self.seq_len)
                noise = rng.normal(0, SYNTHETIC_NOISE_STD, self.seq_len)
                periodic = SYNTHETIC_PERIODIC_AMPLITUDE * np.sin(
                    np.linspace(0, SYNTHETIC_PERIODIC_CYCLES * np.pi, self.seq_len)
                    + rng.uniform(0, 2 * np.pi)
                )

                values = baseline + np.cumsum(trend) + noise + periodic
                sample[:, feat_idx] = torch.from_numpy(np.clip(values, 0, 1)).float()

            # Generate synthetic target based on task type
            if self.task_type == "regression":
                # For regression: predict some function of the sequence statistics
                # E.g., predict the mean of the last portion of the sequence
                last_portion_start = int(SYNTHETIC_TARGET_PORTION * self.seq_len)
                last_portion = sample[last_portion_start:, :]
                target = torch.mean(last_portion, dim=(0, 1)).unsqueeze(
                    0
                )  # Shape: (1,)

                if self.prediction_dim > 1:
                    # For multi-dimensional regression, create multiple correlated targets
                    target = target.repeat(self.prediction_dim)
                    noise = torch.from_numpy(
                        rng.normal(
                            0, SYNTHETIC_REGRESSION_NOISE_STD, self.prediction_dim
                        )
                    ).float()
                    target += noise

            elif self.task_type == "classification":
                # For classification: create synthetic classes based on sequence properties
                # E.g., classify based on whether mean value is above/below threshold
                mean_val = torch.mean(sample)
                if self.prediction_dim == 1:
                    # Binary classification (but still need class index, not probability)
                    target = torch.tensor(
                        [1 if mean_val > SYNTHETIC_CLASSIFICATION_THRESHOLD else 0],
                        dtype=torch.long,
                    )
                else:
                    # Multi-class classification
                    # Divide the [0,1] range into prediction_dim bins
                    bin_size = 1.0 / self.prediction_dim
                    class_idx = min(int(mean_val / bin_size), self.prediction_dim - 1)
                    target = torch.tensor([class_idx], dtype=torch.long)
            else:
                raise ValueError(f"Unsupported task_type: {self.task_type}")

            yield sample, target


class FinetunedDataloader(DataLoader):
    """DataLoader wrapper for synthetic fine-tuning data"""

    def __init__(
        self,
        batch_size: int,
        split: str,
        shuffle: bool,
        task_type: str,
        prediction_dim: int,
        world_size: int,
        rank: int,
    ):
        dataset = SyntheticFinetunedDataset(
            split=split, rank=rank, task_type=task_type, prediction_dim=prediction_dim
        )

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        self.task_type = task_type
        self.prediction_dim = prediction_dim
        self.device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

    def __iter__(self):
        """Yield batches in format: (input_tensor, target_tensor)"""
        iteration_count = 0
        for batch_data in super().__iter__():
            # Limit iterations in dry run mode (using same limit as pretraining)
            if DRY_RUN and iteration_count >= DRY_RUN_ITERATIONS:
                break

                # batch_data is what the DataLoader returns - already batched tensors
            input_tensor, target_tensor = batch_data

            # Move to device
            input_tensor = input_tensor.to(self.device)
            target_tensor = target_tensor.to(self.device)

            # For classification, squeeze the extra dimension
            if self.task_type == "classification":
                target_tensor = target_tensor.squeeze(-1)

            # Create empty input_feature_mask (no masking for finetuning)
            batch_size, seq_len, n_features = input_tensor.shape
            input_feature_mask = torch.zeros(
                batch_size, seq_len, n_features, dtype=torch.bool, device=self.device
            )

            # No padding mask needed since all sequences are same length
            src_key_padding_mask = None

            yield input_tensor, input_feature_mask, target_tensor, src_key_padding_mask
            iteration_count += 1
