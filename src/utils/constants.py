import os
import dotenv

dotenv.load_dotenv()

DATA_DIR = os.environ.get("DATA_DIR", "data")
DRY_RUN = os.environ.get("DRY_RUN", "True").lower() in ["true", "1", "t", "y", "yes"]
DRY_RUN_ITERATIONS = 10
MAX_CONTEXT_LENGTH = 1024
INPUT_DIM = 32  # number of input features
OUTPUT_DIM = 32  # number of output features


# variance range (clamp to prevent instability)
VAR_MIN = 1e-8
VAR_MAX = 1

# Fine-tuning synthetic data generation constants
SYNTHETIC_BASELINE_MIN = 0.3
SYNTHETIC_BASELINE_RANGE = 0.4
SYNTHETIC_TREND_STD = 0.01
SYNTHETIC_NOISE_STD = 0.05
SYNTHETIC_PERIODIC_AMPLITUDE = 0.1
SYNTHETIC_PERIODIC_CYCLES = 4
SYNTHETIC_TARGET_PORTION = 0.9  # Use last 10% of sequence for regression targets
SYNTHETIC_REGRESSION_NOISE_STD = 0.1
SYNTHETIC_CLASSIFICATION_THRESHOLD = 0.5

# Trainer constants
DEFAULT_RANK = 0
DEFAULT_WORLD_SIZE = 1
DEFAULT_LOCAL_RANK = 0
SQUEEZE_LAST_DIM = -1
TRAIN_SPLIT = "train"
VAL_SPLIT = "validation"
