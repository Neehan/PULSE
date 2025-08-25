# PULSE - Probabilistic Unsupervised Latent Sequence Encoder

PULSE is a deep learning framework for pretraining autoencoder models on time series data, specifically designed for ICU vitals prediction tasks.

## Setup

### 1. Create Conda Environment

First, create a new conda environment with Python:

```bash
conda create -n pulse python=3.11
conda activate pulse
```

### 2. Install Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the project root:

```bash
DRY_RUN=True
DATA_DIR=data/
```

## Usage

### Pretraining

Run pretraining with default parameters:

```bash
python -m src.pretraining.pretraining_main
```

### Model Sizes

Available model sizes:
- `mini` (59k parameters) - Fast training, good for testing
- `small` (1.8M parameters) - Default size
- `medium` (13M parameters) - Larger model
- `large` (56M parameters) - Largest model

Example with mini model:

```bash
PYTHONPATH=. python -m src.pretraining.pretraining_main --model-size mini
```

### Training Parameters

Key training parameters you can adjust:

```bash
PYTHONPATH=. python -m src.pretraining.pretraining_main \
    --model-size mini \
    --batch-size 256 \
    --n-epochs 50 \
    --init-lr 0.0005 \
```

## Project Structure

```
PULSE/
├── src/
│   ├── base/                    # Base classes
│   │   ├── base_model.py       # Base model class
│   │   └── base_trainer.py     # Base trainer class
│   ├── models/                 # Model implementations
│   │   ├── autoencoder.py      # Autoencoder model
│   │   ├── pulse_normal.py     # PULSE normal variant
│   │   └── pulse_sinusoid.py   # PULSE sinusoidal variant
│   ├── pretraining/            # Pretraining components
│   │   ├── dataloaders/        # Data loading utilities
│   │   ├── trainers/           # Training logic
│   │   └── pretraining_main.py # Main pretraining script
│   └── utils/                  # Utility functions
│       ├── constants.py        # Configuration constants
│       ├── pretraining_masks.py # Masking functions
│       └── utils.py           # General utilities
├── data/                       # Data directory
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## Features

- **Autoencoder Pretraining**: Masked time series prediction
- **Multiple Model Sizes**: From mini (59k) to large (56M) parameters
- **Flexible Training**: Configurable learning rates, schedulers, and early stopping
- **Distributed Training Support**: Multi-GPU training capabilities
- **Dry Run Mode**: Test without actual training for debugging

## Environment Variables

- `DRY_RUN`: Set to `True` for testing without actual training
- `DATA_DIR`: Directory for data and model outputs (default: `data/`)

## Development

The project follows a modular architecture:

1. **Base Classes**: `BaseModel` and `BaseTrainer` provide common functionality
2. **Models**: Specific model implementations inherit from `BaseModel`
3. **Trainers**: Training logic specific to each model type
4. **Utils**: Shared utilities for data processing, masking, and configuration

## License

See LICENSE file for details.
