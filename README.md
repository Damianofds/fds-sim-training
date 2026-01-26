# SIM v0 Training Code

Spatial Intent Model (SIM) v0 training code with frozen LLM for autonomous rover navigation.

## Overview

This repository implements a multimodal model that combines video and telemetry data to predict:
- **Risk level** (none, low, medium, high)
- **Reason** (human, obstacle, tight_space, unknown)
- **Action** (continue, slow, stop, reverse, turn_left, turn_right)

### Architecture

- **Frozen LLM backbone** (e.g., GPT-2, OPT) - parameters not updated during training
- **Frozen vision encoder** (optional, for raw frames)
- **Trainable components**:
  - Vision projector: maps vision features to LLM embedding space
  - Telemetry MLP: maps 8D telemetry to LLM embedding space
  - Task query token: learnable token for readout
  - Classification heads: risk (4 classes), reason (4 classes), action (6 classes)

### Key Features

- Config-driven training pipeline
- Automatic mixed precision (AMP) support
- Comprehensive metrics logging
- Checkpoint management (best/last)
- Support for both raw frames and cached features
- Reproducible training with seed control

## Installation

### 1. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Run training with default config

```bash
python train_sim_v0.py --config configs/sim_v0.yaml
```

This will:
1. Load the dummy dataset (random data for testing)
2. Initialize a GPT-2 based model
3. Train for 10 epochs
4. Save checkpoints to `outputs/sim_v0_run/`

## Configuration

Edit `configs/sim_v0.yaml` to customize training. Key parameters:

### Data Configuration

```yaml
data:
  dataset_class_path: "sim.dataset.DummySimDataset"  # Dataset class
  train_split_path: "data/train"  # Path to training data
  val_split_path: "data/val"      # Path to validation data
  batch_size: 32
  num_workers: 0
```

### Model Configuration

```yaml
model:
  llm_name_or_path: "gpt2"        # HuggingFace model name
  use_cached_vis_feats: true      # Use cached features (faster)
  dv: 768   # Vision feature dimension
  d: 768    # LLM embedding dimension
  F: 8      # Frames per sample
  K: 4      # Visual tokens after pooling
```

### Training Configuration

```yaml
training:
  epochs: 10
  lr: 0.001
  weight_decay: 0.01
  grad_clip: 1.0
  amp: false  # Automatic mixed precision
  seed: 42
```

## Custom Dataset

To use your own dataset, create a subclass of `SimDataset`:

```python
from sim.dataset import SimDataset
import torch

class MyRoverDataset(SimDataset):
    def __init__(self, data_path: str, use_cached_vis_feats: bool = True):
        super().__init__(data_path, use_cached_vis_feats)
        # Load your data here
        self.samples = self.load_data(data_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "vis_feats": torch.tensor(sample["vis_feats"]),  # [F, dv]
            "telemetry": torch.tensor(sample["telemetry"]),  # [8]
            "risk_id": sample["risk_id"],      # int 0-3
            "reason_id": sample["reason_id"],  # int 0-3
            "action_id": sample["action_id"],  # int 0-5
            "sample_id": sample["id"],         # str (optional)
        }
```

Then update your config:

```yaml
data:
  dataset_class_path: "my_module.MyRoverDataset"
  train_split_path: "path/to/train.pkl"
  val_split_path: "path/to/val.pkl"
```

## Input Data Format

Your dataset must return dictionaries with:

### Video (choose one):
- **Option A**: Cached features (recommended for speed)
  - `"vis_feats"`: `torch.Tensor [F, dv]`
- **Option B**: Raw frames
  - `"frames"`: `torch.Tensor [F, 3, H, W]`

### Telemetry:
- `"telemetry"`: `torch.Tensor [8]` - `[ax, ay, az, gx, gy, gz, v, w]`
  - Normalized (mean 0, std 1) before training

### Labels:
- `"risk_id"`: `int` in {0, 1, 2, 3} - (none, low, medium, high)
- `"reason_id"`: `int` in {0, 1, 2, 3} - (human, obstacle, tight_space, unknown)
- `"action_id"`: `int` in {0, 1, 2, 3, 4, 5} - (continue, slow, stop, reverse, turn_left, turn_right)

### Optional:
- `"weight"`: `float` - Sample weight for class imbalance
- `"sample_id"`: `str` - For debugging

## Output Structure

Training outputs are saved to the specified `output_dir`:

```
outputs/sim_v0_run/
├── config_resolved.yaml          # Saved configuration
├── checkpoints/
│   ├── best.pt                   # Best model (by risk_high_recall)
│   ├── last.pt                   # Most recent checkpoint
│   └── epoch_001.pt              # Per-epoch checkpoints (optional)
├── metrics/
│   ├── train_metrics.jsonl       # Training metrics per epoch
│   └── val_metrics.jsonl         # Validation metrics per epoch
├── logs/
│   └── stdout.log                # Training logs
└── export/
    └── labels.json               # Label ID to name mappings
```

### Checkpoint Contents

Each checkpoint (`.pt` file) contains:
- `model_state_dict`: Trainable module weights
- `optimizer_state_dict`: Optimizer state
- `epoch`: Epoch number
- `metrics`: Validation metrics
- `metadata`: Model architecture parameters and label maps

## Key Metrics

The training code tracks:

- **risk_high_recall**: Recall for "high" risk class (critical safety metric)
- **risk_high_fn_rate**: False negative rate for "high" risk (1 - recall)
- **action_accuracy**: Overall action prediction accuracy
- **macro_f1**: Macro-averaged F1 score (risk + action)
- **risk_f1_macro**: Risk classification F1
- **reason_f1_macro**: Reason classification F1

Best model is selected based on `risk_high_recall`.

## Loss Function

```
L = CE(risk) + 0.5 * CE(reason) + CE(action)
```

Where CE is cross-entropy loss.

## Loading Checkpoints

```python
from sim.model import SimV0Model
import torch

# Load checkpoint
checkpoint = torch.load("outputs/sim_v0_run/checkpoints/best.pt")

# Create model with same config
model = SimV0Model(
    llm_name_or_path="gpt2",
    dv=768,
    d=768,
    F=8,
    K=4,
    use_cached_vis_feats=True,
)

# Load weights
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Make predictions
batch = {...}  # Your input batch
predictions = model.predict_json(batch)
print(predictions)
```

## Advanced Usage

### Using Different LLMs

You can use any HuggingFace decoder-only model:

```yaml
model:
  llm_name_or_path: "facebook/opt-125m"  # Smaller, faster
  d: 768  # Must match model's embedding dimension
```

Or larger models:

```yaml
model:
  llm_name_or_path: "facebook/opt-1.3b"
  d: 2048
```

### Mixed Precision Training

Enable AMP for faster training on CUDA:

```yaml
training:
  amp: true
```

### Batch Size Tuning

- **Cached features**: batch_size = 32-64
- **Raw frames**: batch_size = 8-16 (more memory intensive)

## Development

### Run tests

```bash
pytest
```

### Format code

```bash
black .
```

## Troubleshooting

### Out of memory error

1. Reduce batch size
2. Use cached features instead of raw frames
3. Use a smaller LLM (e.g., `gpt2` instead of `gpt2-large`)
4. Enable gradient checkpointing (future feature)

### Low accuracy

1. Check data normalization (telemetry should be normalized)
2. Increase training epochs
3. Try different learning rates (1e-4 to 1e-2)
4. Verify label distributions (class imbalance?)

### Training too slow

1. Use cached vision features
2. Enable AMP (`amp: true`)
3. Increase batch size if memory allows
4. Use multiple workers (`num_workers: 4`)

## Citation

If you use this code in your research, please cite:

```
@software{sim_v0_2024,
  title={SIM v0: Spatial Intent Model with Frozen LLM},
  author={Your Name},
  year={2024},
}
```

## License

MIT License (or specify your license)

## Contact

For questions or issues, please contact [your email] or open an issue on GitHub.
