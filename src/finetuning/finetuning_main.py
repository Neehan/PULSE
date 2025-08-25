import argparse
from src.finetuning.trainers.autoencoder_finetuned_trainer import (
    autoencoder_finetuned_training_loop,
)
from src.utils.utils import setup_distributed, cleanup_distributed, setup_logging
from src.utils.utils import parse_args, set_seed

# Default fine-tuning parameters
DEFAULT_BATCH_SIZE = 64  # Smaller batch size for fine-tuning
DEFAULT_N_EPOCHS = 50  # Fewer epochs for fine-tuning
DEFAULT_INIT_LR = 0.0001  # Lower learning rate for fine-tuning
DEFAULT_N_WARMUP_EPOCHS = 5
DEFAULT_DECAY_FACTOR = 0.95
DEFAULT_MODEL_SIZE = "small"
DEFAULT_SEED = 1234
DEFAULT_SCHEDULER_TYPE = "cosine"
DEFAULT_TRAINING_TYPE = "finetuning"
DEFAULT_EARLY_STOPPING_PATIENCE = 10
DEFAULT_EARLY_STOPPING_MIN_DELTA = 1e-4

# Default fine-tuning specific parameters
DEFAULT_PREDICTION_DIM = 1
DEFAULT_ATTENTION_HIDDEN_DIM = 128
DEFAULT_MLP_HIDDEN_DIM = 256
DEFAULT_TASK_TYPE = "regression"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    help="model type (currently only autoencoder_finetuned supported)",
    default="autoencoder_finetuned",
    type=str,
)
parser.add_argument(
    "--pretrained-model-path",
    help="path to pretrained model to load (optional for testing)",
    required=False,
    default=None,
    type=str,
)
parser.add_argument(
    "--resume-from-checkpoint",
    help="path to resume from checkpoint",
    default=None,
    type=str,
)
parser.add_argument(
    "--batch-size", help="batch size", default=DEFAULT_BATCH_SIZE, type=int
)
parser.add_argument(
    "--n-epochs", help="number of epochs", default=DEFAULT_N_EPOCHS, type=int
)
parser.add_argument(
    "--init-lr", help="initial learning rate", default=DEFAULT_INIT_LR, type=float
)
parser.add_argument(
    "--n-warmup-epochs",
    help="number of warmup epochs",
    default=DEFAULT_N_WARMUP_EPOCHS,
    type=int,
)
parser.add_argument(
    "--decay-factor",
    help="decay factor for exponential scheduler",
    default=DEFAULT_DECAY_FACTOR,
    type=float,
)
parser.add_argument(
    "--model-size",
    help="model size (mini, small, medium, large)",
    default=DEFAULT_MODEL_SIZE,
    type=str,
)
parser.add_argument(
    "--scheduler-type",
    help="scheduler type (cosine or exponential)",
    default=DEFAULT_SCHEDULER_TYPE,
    type=str,
)
parser.add_argument("--seed", help="random seed", default=DEFAULT_SEED, type=int)
parser.add_argument(
    "--early-stopping-patience",
    help="early stopping patience",
    default=DEFAULT_EARLY_STOPPING_PATIENCE,
    type=int,
)
parser.add_argument(
    "--early-stopping-min-delta",
    help="early stopping minimum delta",
    default=DEFAULT_EARLY_STOPPING_MIN_DELTA,
    type=float,
)

# Fine-tuning specific arguments
parser.add_argument(
    "--task-type",
    help="task type (regression or classification)",
    default=DEFAULT_TASK_TYPE,
    type=str,
)
parser.add_argument(
    "--prediction-dim",
    help="prediction dimension (1 for regression, num_classes for classification)",
    default=DEFAULT_PREDICTION_DIM,
    type=int,
)
parser.add_argument(
    "--attention-hidden-dim",
    help="hidden dimension for attention MLP",
    default=DEFAULT_ATTENTION_HIDDEN_DIM,
    type=int,
)
parser.add_argument(
    "--mlp-hidden-dim",
    help="hidden dimension for prediction MLP",
    default=DEFAULT_MLP_HIDDEN_DIM,
    type=int,
)


def main():
    args = parser.parse_args()

    # Validate arguments
    if args.task_type not in ["regression", "classification"]:
        raise ValueError("task_type must be 'regression' or 'classification'")

    if args.model not in ["autoencoder_finetuned"]:
        raise ValueError("Currently only 'autoencoder_finetuned' model is supported")

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()

    # Setup logging and seed
    setup_logging(rank)
    set_seed(args.seed)

    # Parse arguments into dictionary
    args_dict = parse_args(parser)

    # Add fine-tuning specific parameters
    args_dict.update(
        {
            "task_type": args.task_type,
            "prediction_dim": args.prediction_dim,
            "attention_hidden_dim": args.attention_hidden_dim,
            "mlp_hidden_dim": args.mlp_hidden_dim,
            "rank": rank,
            "world_size": world_size,
            "local_rank": local_rank,
        }
    )

    model_type = args.model

    try:
        if model_type == "autoencoder_finetuned":
            best_val_loss = autoencoder_finetuned_training_loop(args_dict)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        if rank == 0:
            print(f"Fine-tuning completed. Best validation loss: {best_val_loss:.4f}")

    except Exception as e:
        if rank == 0:
            print(f"Fine-tuning failed with error: {e}")
        raise e
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
