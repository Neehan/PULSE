import torch


def full_feature_mask_random(
    seq_len: int, batch_size: int, input_dim: int, n_masked_features: int, device: str
):
    """
    Randomly masks n_masked_features completely across all timesteps.
    Args:
        seq_len: length of the sequence
        batch_size: number of samples in the batch
        input_dim: number of input features
        n_masked_features: number of features to mask
        device: device to use for the mask

    Returns:
        mask: [batch_size, seq_len, input_dim]
    """
    # Generate random values and sort to get permutations: [batch_size, input_dim]
    random_vals = torch.rand(batch_size, input_dim, device=device)
    rand_perm = torch.argsort(random_vals, dim=-1)

    # Create feature mask: True where perm value < n_masked_features
    # This selects exactly n_masked_features random features per sample
    feature_mask = rand_perm < n_masked_features  # [batch_size, input_dim]

    # Expand to all timesteps: [batch_size, seq_len, input_dim]
    mask = feature_mask.unsqueeze(1).expand(-1, seq_len, -1)

    return mask
