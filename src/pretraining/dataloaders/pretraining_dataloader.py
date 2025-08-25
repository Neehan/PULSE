import torch
import numpy as np
from torch.utils.data import DataLoader, IterableDataset
from src.utils.constants import INPUT_DIM, MAX_CONTEXT_LENGTH, DRY_RUN
from typing import Callable


class SyntheticDataset(IterableDataset):
    """Generates synthetic ICU vitals on-the-fly"""

    def __init__(
        self, split: str, rank: int, masking_function: Callable, n_masked_features: int
    ):
        self.masking_function = masking_function
        self.n_masked_features = n_masked_features
        self.device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        self.seq_len = min(512, MAX_CONTEXT_LENGTH)
        self.input_dim = INPUT_DIM

    def __iter__(self):
        """Generate infinite stream of synthetic samples"""
        while True:
            # Generate one sample of synthetic ICU vitals
            sample = torch.zeros(self.seq_len, self.input_dim)

            for feat_idx in range(self.input_dim):
                # Simple realistic time series: baseline + trend + noise + periodic
                baseline = 0.3 + 0.4 * np.random.random()
                trend = np.random.normal(0, 0.01, self.seq_len)
                noise = np.random.normal(0, 0.05, self.seq_len)
                periodic = 0.1 * np.sin(
                    np.linspace(0, 4 * np.pi, self.seq_len)
                    + np.random.uniform(0, 2 * np.pi)
                )

                values = baseline + np.cumsum(trend) + noise + periodic
                sample[:, feat_idx] = torch.from_numpy(np.clip(values, 0, 1)).float()

            yield sample


class PretrainingDataloader(DataLoader):
    """DataLoader wrapper for synthetic data that applies masking"""

    def __init__(
        self,
        batch_size: int,
        split: str,
        shuffle: bool,
        masking_function: Callable,
        n_masked_features: int,
        world_size: int,
        rank: int,
    ):
        dataset = SyntheticDataset(split, rank, masking_function, n_masked_features)

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        self.masking_function = masking_function
        self.n_masked_features = n_masked_features
        self.device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

    def __iter__(self):
        """Yield batches in format: (input_tensor, input_feature_mask, src_key_padding_mask)"""
        iteration_count = 0
        for batch in super().__iter__():
            # Limit to 10 iterations in dry run mode
            if DRY_RUN and iteration_count >= 10:
                break

            batch = batch.to(self.device)
            batch_size, seq_len, input_dim = batch.shape

            # Generate mask if masking function provided
            if self.masking_function:
                input_feature_mask = self.masking_function(
                    seq_len=seq_len,
                    batch_size=batch_size,
                    input_dim=input_dim,
                    n_masked_features=self.n_masked_features,
                    device=self.device,
                )
            else:
                input_feature_mask = torch.zeros(
                    batch_size, seq_len, input_dim, dtype=torch.bool, device=self.device
                )

            # No padding mask needed since all sequences are same length
            src_key_padding_mask = None

            yield batch, input_feature_mask, src_key_padding_mask
            iteration_count += 1
