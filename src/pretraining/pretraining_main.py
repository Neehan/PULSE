import argparse
from src.pretraining.trainers.autoencoder_trainer import autoencoder_training_loop
from src.pretraining.trainers.pulse_normal_trainer import pulse_normal_training_loop
from src.pretraining.trainers.pulse_sinusoid_trainer import pulse_sinusoid_training_loop
from src.utils.utils import setup_distributed, cleanup_distributed, setup_logging
from src.utils.utils import parse_args, set_seed
from src.utils.enums import SchedulerType

# Default training parameters - all magic numbers defined here
DEFAULT_BATCH_SIZE = 256
DEFAULT_N_EPOCHS = 100
DEFAULT_INIT_LR = 0.0005
DEFAULT_N_WARMUP_EPOCHS = 10
DEFAULT_DECAY_FACTOR = 0.99
DEFAULT_MODEL_SIZE = "small"
DEFAULT_SEED = 1234
# Default autoencoder-specific parameters
DEFAULT_INITIAL_N_MASKED_FEATURES = 5
DEFAULT_MAX_N_MASKED_FEATURES = 24
DEFAULT_N_MASKED_FEATURES_INCREASE_EVERY_N_EPOCHS = 5
DEFAULT_SCHEDULER_TYPE = SchedulerType.EXPONENTIAL.value
DEFAULT_TRAINING_TYPE = "pretraining"
DEFAULT_EARLY_STOPPING_PATIENCE = 15
DEFAULT_EARLY_STOPPING_MIN_DELTA = 1e-3
# Default pulse_normal and pulse_sinusoid-specific parameters
DEFAULT_ALPHA = 0.5
DEFAULT_K_COMPONENTS = 1

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    help="model type (autoencoder, pulse_normal, pulse_sinusoid)",
    default="autoencoder",
    type=str,
)
parser.add_argument(
    "--resume-from-checkpoint",
    help="path to resume from checkpoint",
    default=None,
    type=str,
)
parser.add_argument(
    "--pretrained-model-path",
    help="path to pretrained model to load before training",
    default=None,
    type=str,
)
parser.add_argument(
    "--batch-size", help="batch size", default=DEFAULT_BATCH_SIZE, type=int
)
parser.add_argument(
    "--n-epochs", help="number of training epochs", default=DEFAULT_N_EPOCHS, type=int
)
parser.add_argument(
    "--init-lr", help="initial learning rate", default=DEFAULT_INIT_LR, type=float
)
parser.add_argument(
    "--n-warmup-epochs",
    help="number of warm-up epochs",
    default=DEFAULT_N_WARMUP_EPOCHS,
    type=float,
)
parser.add_argument(
    "--decay-factor",
    help="exponential learning rate decay factor after warmup",
    default=DEFAULT_DECAY_FACTOR,
    type=float,
)
parser.add_argument(
    "--model-size",
    help="model size mini (59k), small (1.8M), medium (13M), and large (56M)",
    default=DEFAULT_MODEL_SIZE,
    type=str,
)
parser.add_argument(
    "--initial-n-masked-features",
    help="initial number of masked features for autoencoder training",
    default=DEFAULT_INITIAL_N_MASKED_FEATURES,
    type=int,
)
parser.add_argument(
    "--max-n-masked-features",
    help="maximum number of masked features for autoencoder training",
    default=DEFAULT_MAX_N_MASKED_FEATURES,
    type=int,
)
parser.add_argument(
    "--n-masked-features-increase-every-n-epochs",
    help="increase masked features every N epochs",
    default=DEFAULT_N_MASKED_FEATURES_INCREASE_EVERY_N_EPOCHS,
    type=int,
)
parser.add_argument(
    "--scheduler-type",
    help="type of learning rate scheduler (cosine, exponential, or linear_flat)",
    default=DEFAULT_SCHEDULER_TYPE,
    type=str,
)
parser.add_argument(
    "--early-stopping-patience",
    help="number of epochs to wait before early stopping",
    default=DEFAULT_EARLY_STOPPING_PATIENCE,
    type=int,
)
parser.add_argument(
    "--early-stopping-min-delta",
    help="minimum change in validation loss for early stopping",
    default=DEFAULT_EARLY_STOPPING_MIN_DELTA,
    type=float,
)
parser.add_argument(
    "--seed",
    help="random seed for reproducibility",
    default=DEFAULT_SEED,
    type=int,
)
parser.add_argument(
    "--alpha",
    help="alpha parameter for VAE loss weighting (pulse_normal and pulse_sinusoid only)",
    default=DEFAULT_ALPHA,
    type=float,
)
parser.add_argument(
    "--k-components",
    help="number of sinusoidal components for pulse_sinusoid model",
    default=DEFAULT_K_COMPONENTS,
    type=int,
)


def main():
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()

    # Setup logging
    setup_logging(rank)

    try:
        args_dict = parse_args(parser)

        # Set seed for reproducibility (rank-aware for distributed training)
        set_seed(args_dict["seed"] + rank)

        # Add distributed training info to args
        args_dict["rank"] = rank
        args_dict["world_size"] = world_size
        args_dict["local_rank"] = local_rank

        # Set training type since we're in pretraining
        args_dict["training_type"] = DEFAULT_TRAINING_TYPE

        model_type = args_dict["model"].lower()

        if model_type == "autoencoder":
            autoencoder_training_loop(args_dict)
        elif model_type == "pulse_normal":
            pulse_normal_training_loop(args_dict)
        elif model_type == "pulse_sinusoid":
            pulse_sinusoid_training_loop(args_dict)
        else:
            raise ValueError(
                f"Unknown model type: {model_type}. Supported types: 'autoencoder', 'pulse_normal', 'pulse_sinusoid'"
            )
    finally:
        # Clean up distributed environment
        cleanup_distributed()


if __name__ == "__main__":
    main()
