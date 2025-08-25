import os
import dotenv

dotenv.load_dotenv()

DATA_DIR = os.environ.get("DATA_DIR", "data")
DRY_RUN = os.environ.get("DRY_RUN", "True").lower() in ["true", "1", "t", "y", "yes"]
MAX_CONTEXT_LENGTH = 1024
INPUT_DIM = 32  # number of input features
OUTPUT_DIM = 32  # number of output features


# variance range (clamp to prevent instability)
VAR_MIN = 1e-8
VAR_MAX = 1
