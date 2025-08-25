# PULSE - Probabilistic Unsupervised Latent Sequence Encoder

PULSE is a deep learning framework for pretraining transformer-based models on time series data, specifically designed for ICU vitals prediction tasks. PULSE models use variational autoencoders with probabilistic outputs, while the autoencoder serves as a baseline for comparison.

## Quick Start

### 1. Installation

Create a conda environment and install dependencies:

```bash
# Create environment
conda create -n pulse python=3.11
conda activate pulse

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file in the project root:

```bash
DRY_RUN=True
DATA_DIR=data/
```

- `DRY_RUN=True`: Limits training to 10 iterations for quick testing
- `DRY_RUN=False`: Full training mode
- `DATA_DIR`: Directory for model outputs and data

### 3. Run Pretraining

Train a PULSE model with default settings:

```bash
# PULSE Normal model (default)
python -m src.pretraining.pretraining_main --model pulse_normal

# PULSE Sinusoid model (with sinusoidal priors)
python -m src.pretraining.pretraining_main --model pulse_sinusoid

# Autoencoder baseline
python -m src.pretraining.pretraining_main --model autoencoder
```

### 4. Run Fine-tuning 

After pretraining, fine-tune for specific prediction tasks:

```bash
# Fine-tune for regression task
python -m src.finetuning.finetuning_main \
    --task-type regression \
    --pretrained-model-path data/trained_models/pretrained_autoencoder.pth

# Fine-tune for classification task  
python -m src.finetuning.finetuning_main \
    --task-type classification \
    --prediction-dim 3 \
    --pretrained-model-path data/trained_models/pretrained_autoencoder.pth
```

## Training

PULSE has two training modes:
- **Pretraining**: Learn representations from time series data
- **Fine-tuning**: Add prediction head for specific tasks

## Model Architecture

### PULSE Models (Main Framework)

PULSE models are variational autoencoders that output probabilistic predictions:

- **PULSE Normal**: Outputs Gaussian distributions (μ, σ²) for each predicted value
- **PULSE Sinusoid**: Extends PULSE Normal with learnable sinusoidal priors for capturing periodic patterns in ICU vitals

### Autoencoder (Baseline)

The autoencoder provides deterministic point predictions and serves as a baseline for comparison with PULSE models.

## Configuration

### Model Sizes
- `mini` (59k params) - Fast training
- `small` (1.8M params) - Default 
- `medium` (13M params) - More capacity
- `large` (56M params) - Maximum capacity

```bash
python -m src.pretraining.pretraining_main --model pulse_normal --model-size large
```

## Data

Uses synthetic ICU vitals with 32 features and up to 512 timesteps.

## How to Add a New Model

### 1. Create Model Class

Create a new model file in `src/models/`:

```python
# src/models/your_new_model.py
import torch
import torch.nn as nn
from src.base.base_model import BaseModel
from src.utils.constants import MAX_CONTEXT_LENGTH

class YourNewModel(BaseModel):
    def __init__(self, input_dim: int, output_dim: int, device: str, ...):
        super().__init__(name="your_new_model")
        # Define your architecture here
        
    def load_pretrained(self, pretrained_model: "YourNewModel"):
        # Implement pretrained model loading
        pass
        
    def forward(self, input_tensor, input_feature_mask, src_key_padding_mask=None):
        # Implement forward pass
        pass
```

### 2. Create Trainer Class

Create a trainer in `src/pretraining/trainers/`:

```python
# src/pretraining/trainers/your_new_model_trainer.py
from src.base.base_trainer import BaseTrainer
from src.models.your_new_model import YourNewModel

class YourNewModelTrainer(BaseTrainer):
    def get_dataloaders(self):
        # Return train and validation dataloaders
        pass
        
    def compute_train_loss(self, input_tensor, input_feature_mask, src_key_padding_mask):
        # Implement training loss computation
        pass
        
    def compute_validation_loss(self, input_tensor, input_feature_mask, src_key_padding_mask):
        # Implement validation loss computation
        pass

def your_new_model_training_loop(args_dict):
    # Initialize model and trainer
    # Return trainer.train()
    pass
```

### 3. Register in Main Script

Add your model to `src/pretraining/pretraining_main.py`:

```python
# Add import
from src.pretraining.trainers.your_new_model_trainer import your_new_model_training_loop

# Add to argument parser help text
parser.add_argument(
    "--model",
    help="model type (autoencoder, pulse_normal, pulse_sinusoid, your_new_model)",
    # ...
)

# Add to main() function
elif model_type == "your_new_model":
    your_new_model_training_loop(args_dict)
```

## How to Add a New Dataloader

### 1. Create Dataset Class

Create a new dataset in `src/pretraining/dataloaders/`:

```python
# src/pretraining/dataloaders/your_dataloader.py
import torch
from torch.utils.data import Dataset, IterableDataset

class YourDataset(Dataset):  # or IterableDataset for streaming
    def __init__(self, split: str, ...):
        # Initialize your dataset
        pass
        
    def __len__(self):
        # Return dataset size (for Dataset, not IterableDataset)
        pass
        
    def __getitem__(self, idx):
        # Return single sample: torch.Tensor of shape (seq_len, input_dim)
        pass
```

### 2. Create DataLoader Wrapper

```python
class YourDataLoader(DataLoader):
    def __init__(self, batch_size: int, split: str, ...):
        dataset = YourDataset(split, ...)
        super().__init__(dataset=dataset, batch_size=batch_size, ...)
        
    def __iter__(self):
        for batch in super().__iter__():
            # Apply any preprocessing/masking
            # Yield: (input_tensor, input_feature_mask, src_key_padding_mask)
            yield batch, input_feature_mask, src_key_padding_mask
```

### 3. Update Trainer

Modify your trainer's `get_dataloaders()` method:

```python
def get_dataloaders(self):
    train_loader = YourDataLoader(
        batch_size=self.batch_size,
        split="train",
        # ... other parameters
    )
    val_loader = YourDataLoader(
        batch_size=self.batch_size,
        split="validation",
        # ... other parameters
    )
    return train_loader, val_loader
```

## Advanced Usage

### Distributed Training

The framework supports multi-GPU training automatically:

```bash
# Single GPU
python -m src.pretraining.pretraining_main --model pulse_normal

# Multi-GPU (automatically detected)
torchrun --nproc_per_node=4 -m src.pretraining.pretraining_main --model pulse_normal
```

### Multi-GPU Training

```bash
torchrun --nproc_per_node=4 -m src.pretraining.pretraining_main --model pulse_normal
```

## Project Structure

```
PULSE/
├── src/
│   ├── base/                           # Base classes
│   │   ├── base_model.py              # BaseModel - inherit for new models
│   │   └── base_trainer.py            # BaseTrainer - inherit for new trainers
│   ├── models/                        # Model implementations
│   │   ├── autoencoder.py             # Autoencoder baseline model
│   │   ├── pulse_normal.py            # PULSE Normal (Gaussian VAE)
│   │   ├── pulse_sinusoid.py          # PULSE Sinusoid (with sinusoidal priors)
│   │   └── vanilla_pos_encoding.py    # Positional encoding utility
│   ├── pretraining/                   # Pretraining components
│   │   ├── dataloaders/               # Data loading utilities
│   │   │   └── pretraining_dataloader.py  # Synthetic ICU vitals generator
│   │   ├── trainers/                  # Training logic for each model
│   │   │   ├── autoencoder_trainer.py     # Autoencoder training
│   │   │   ├── pulse_normal_trainer.py    # PULSE Normal training
│   │   │   └── pulse_sinusoid_trainer.py  # PULSE Sinusoid training
│   │   └── pretraining_main.py        # Main training script
│   ├── finetuning/                    # Fine-tuning components
│   │   ├── dataloaders/               # Fine-tuning data utilities
│   │   │   └── finetuning_dataloader.py   # Synthetic regression/classification data
│   │   ├── models/                    # Fine-tuned model implementations
│   │   │   └── autoencoder_finetuned.py   # Autoencoder with predictor head
│   │   ├── trainers/                  # Fine-tuning training logic
│   │   │   └── autoencoder_finetuned_trainer.py  # Fine-tuning trainer
│   │   └── finetuning_main.py         # Main fine-tuning script
│   └── utils/                         # Utility functions
│       ├── constants.py               # Configuration constants
│       ├── losses.py                  # Loss functions
│       ├── pretraining_masks.py       # Masking strategies
│       └── utils.py                   # General utilities
├── data/                              # Output directory for models/logs
├── datatrained_models/                # Pretrained model storage
│   └── pretraining/                   # Pretraining checkpoints
├── requirements.txt                   # Python dependencies
└── README.md                         # This guide
```

## Key Concepts

### Progressive Masking

Training uses progressive masking where the number of masked features increases over time:
- Starts with `--initial-n-masked-features` (default: 5)
- Increases every `--n-masked-features-increase-every-n-epochs` (default: 5)
- Caps at `--max-n-masked-features` (default: 24)

### ELBO Loss (PULSE Models)

PULSE models use Evidence Lower BOund (ELBO) loss combining:
- **Reconstruction loss**: How well the model predicts masked values
- **KL divergence**: Regularization term for the variational distribution
- **Alpha weighting**: Balance between reconstruction and regularization (`--alpha`)

### Model Hierarchy

```
BaseModel (abstract)
├── Autoencoder (deterministic baseline)
└── PULSENormal (probabilistic, Gaussian outputs)
    └── PULSESinusoid (adds sinusoidal priors)
```

## Development Tips

1. **Start with mini model**: Use `--model-size mini` for fast iteration
2. **Use dry run**: Keep `DRY_RUN=True` during development
3. **Monitor logs**: Check `data/logs/` for training progress
4. **Checkpoints**: Models automatically save to `data/trained_models/`

## Fine-tuning

Fine-tune pretrained models for prediction tasks:

```bash
# Regression
python -m src.finetuning.finetuning_main \
    --task-type regression \
    --pretrained-model-path data/trained_models/pretrained_autoencoder.pth

# Classification  
python -m src.finetuning.finetuning_main \
    --task-type classification \
    --prediction-dim 3 \
    --pretrained-model-path data/trained_models/pretrained_autoencoder.pth
```

## License

See LICENSE file for details.
