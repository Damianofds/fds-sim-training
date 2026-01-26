# Project Structure

Complete overview of the SIM v0 training codebase.

## Directory Layout

```
fds-sim-training/
├── sim/                          # Core package
│   ├── __init__.py               # Package initialization
│   ├── model.py                  # SimV0Model architecture
│   ├── dataset.py                # Dataset base class and dummy dataset
│   ├── training.py               # Training and evaluation loops
│   ├── metrics.py                # Metrics computation
│   ├── utils.py                  # Utilities (seeding, checkpointing, etc.)
│   └── inference.py              # Inference wrapper
│
├── configs/                      # Configuration files
│   └── sim_v0.yaml               # Default training config
│
├── train_sim_v0.py               # Main training script
├── test_sim.py                   # Test suite
├── example_custom_dataset.py     # Custom dataset template
├── example_training.sh           # Example training script
│
├── requirements.txt              # Python dependencies
├── README.md                     # Full documentation
├── QUICKSTART.md                 # Quick start guide
├── PROJECT_STRUCTURE.md          # This file
├── original-specs.md             # Original specifications
└── .gitignore                    # Git ignore patterns
```

## Core Files

### `sim/model.py`
**SimV0Model architecture**

Key components:
- `VisionProjector`: Projects vision features to LLM embedding space
- `TelemetryMLP`: Projects telemetry to LLM embedding space
- `TemporalPooling`: Reduces F frames to K tokens via segment averaging
- `SimV0Model`: Main model combining frozen LLM with trainable adapters

Methods:
- `forward()`: Forward pass returning logits
- `predict_json()`: Generate human-readable predictions
- `get_metadata()`: Return model metadata for checkpointing

### `sim/dataset.py`
**Dataset interface and implementations**

Classes:
- `SimDataset`: Abstract base class for custom datasets
- `DummySimDataset`: Dummy dataset for testing
- `collate_fn()`: Batching function for DataLoader

Label mappings:
- Risk: {0: none, 1: low, 2: medium, 3: high}
- Reason: {0: human, 1: obstacle, 2: tight_space, 3: unknown}
- Action: {0: continue, 1: slow, 2: stop, 3: reverse, 4: turn_left, 5: turn_right}

### `sim/training.py`
**Training and evaluation loops**

Functions:
- `compute_loss()`: Multi-task loss computation
- `train_one_epoch()`: Training loop with AMP support
- `evaluate()`: Evaluation loop

Loss function:
```
L = CE(risk) + 0.5 * CE(reason) + CE(action)
```

### `sim/metrics.py`
**Metrics computation**

Key metrics:
- `risk_high_recall`: Recall for "high" risk class (most important)
- `risk_high_fn_rate`: False negative rate for "high" risk
- `action_accuracy`: Overall action accuracy
- `macro_f1`: Macro-averaged F1 (risk + action)

Functions:
- `compute_metrics()`: Compute all metrics from predictions
- `print_metrics()`: Pretty print metrics

### `sim/utils.py`
**Utility functions**

Key functions:
- `seed_everything()`: Set random seeds for reproducibility
- `save_checkpoint()` / `load_checkpoint()`: Checkpoint management
- `save_config()` / `load_config()`: Config management
- `setup_output_dir()`: Create output directory structure
- `count_parameters()`: Count trainable/total parameters
- `MetricsLogger`: JSONL metrics logger

### `sim/inference.py`
**Inference wrapper**

Classes:
- `SimV0Inference`: Load checkpoint and run inference

Methods:
- `predict()`: Batch prediction
- `predict_single()`: Single sample prediction
- `get_label_maps()`: Get label mappings

Usage:
```python
sim = SimV0Inference("checkpoints/best.pt", device="cpu")
prediction = sim.predict_single(vis_feats, telemetry)
```

## Configuration

### `configs/sim_v0.yaml`
**Default training configuration**

Sections:
- `data`: Dataset paths, batch size, workers
- `model`: LLM selection, architecture parameters
- `training`: Learning rate, epochs, AMP, seed
- `output`: Output directory

Key parameters:
- `llm_name_or_path`: HuggingFace model (e.g., "gpt2")
- `F`: Number of frames per sample (default: 8)
- `K`: Number of visual tokens (default: 4)
- `dv`: Vision feature dimension (default: 768)
- `d`: LLM embedding dimension (default: 768)

## Scripts

### `train_sim_v0.py`
**Main training script**

Usage:
```bash
python train_sim_v0.py --config configs/sim_v0.yaml
```

Flow:
1. Load configuration
2. Setup seeds and device
3. Create datasets and dataloaders
4. Initialize model and optimizer
5. Training loop with validation
6. Save checkpoints (best/last)
7. Export label maps

### `test_sim.py`
**Comprehensive test suite**

Tests:
1. Model initialization
2. Forward pass
3. Prediction generation
4. Training loop
5. Evaluation
6. Checkpointing
7. Metrics computation

Usage:
```bash
python test_sim.py
```

### `example_custom_dataset.py`
**Template for custom datasets**

Includes:
- `RoverVideoDataset`: Example with cached features
- `RoverRawFramesDataset`: Example with raw frames
- `create_dummy_dataset_files()`: Generate test data

Usage:
```bash
python example_custom_dataset.py
```

## Output Structure

After training, outputs are saved to `output_dir/`:

```
outputs/sim_v0_run/
├── config_resolved.yaml          # Saved configuration
├── checkpoints/
│   ├── best.pt                   # Best model (by risk_high_recall)
│   └── last.pt                   # Latest checkpoint
├── metrics/
│   ├── train_metrics.jsonl       # Training metrics per epoch
│   └── val_metrics.jsonl         # Validation metrics per epoch
├── logs/
│   └── stdout.log                # Training logs (future)
└── export/
    └── labels.json               # Label mappings
```

## Dependencies

### Core ML
- `torch>=2.0.0`: PyTorch
- `torchvision>=0.15.0`: Vision utilities
- `transformers>=4.30.0`: HuggingFace models
- `accelerate>=0.20.0`: Training utilities

### Data & Utils
- `pyyaml>=6.0`: Config parsing
- `numpy>=1.24.0`: Numerical operations
- `pandas>=2.0.0`: Data manipulation

### Metrics & Logging
- `scikit-learn>=1.3.0`: Metrics computation
- `tensorboard>=2.13.0`: Visualization
- `tqdm>=4.65.0`: Progress bars

### Vision (Optional)
- `timm>=0.9.0`: Vision encoders

## Model Architecture

```
Input Batch:
  - vis_feats: [B, F=8, dv=768]
  - telemetry: [B, 8]
    ↓
Vision Projector (trainable):
  - Linear(768 → 768) + GELU + Linear(768 → 768)
  - Output: [B, F=8, d=768]
    ↓
Temporal Pooling (no params):
  - Segment averaging: F=8 → K=4 tokens
  - Output: [B, K=4, d=768]
    ↓
Telemetry MLP (trainable):
  - Linear(8 → 384) + GELU + Linear(384 → 768)
  - Output: [B, 1, d=768]
    ↓
Task Token (trainable):
  - Learned query token
  - Shape: [B, 1, d=768]
    ↓
Concatenate Prefix:
  - [V_tokens(4), T_token(1), Q_token(1)]
  - Shape: [B, L=6, d=768]
    ↓
Frozen LLM (GPT-2):
  - 124M parameters (frozen)
  - Output: [B, L=6, d=768]
    ↓
Readout from Last Token:
  - hidden_states[:, -1, :]
  - Shape: [B, d=768]
    ↓
Classification Heads (trainable):
  - risk_head: Linear(768 → 4)
  - reason_head: Linear(768 → 4)
  - action_head: Linear(768 → 6)
  - confidence_head: Linear(768 → 1)
    ↓
Output Logits:
  - risk_logits: [B, 4]
  - reason_logits: [B, 4]
  - action_logits: [B, 6]
  - confidence: [B, 1]
```

**Trainable Parameters**: ~1.5M
**Frozen Parameters**: ~124M (LLM)
**Total**: ~125M

## Extension Points

### Adding New Metrics
Edit `sim/metrics.py`:
```python
def compute_metrics(...):
    # Add your metric
    custom_metric = my_metric_fn(preds, targets)
    metrics["my_metric"] = custom_metric
    return metrics
```

### Changing Loss Weights
Edit `sim/training.py`:
```python
losses = compute_loss(
    outputs, batch,
    risk_weight=1.0,    # Adjust these
    reason_weight=0.5,
    action_weight=1.0,
)
```

### Using Different LLMs
Edit `configs/sim_v0.yaml`:
```yaml
model:
  llm_name_or_path: "facebook/opt-350m"
  d: 512  # Match LLM embedding dimension
```

### Custom Temporal Pooling
Edit `sim/model.py` `TemporalPooling` class:
```python
def forward(self, x):
    # Implement custom pooling (attention, etc.)
    return pooled_features
```

## Best Practices

1. **Always use cached features** for faster iteration
2. **Start with dummy data** to verify your pipeline
3. **Monitor risk_high_recall** as the key metric
4. **Save checkpoints frequently** in case of crashes
5. **Normalize telemetry** before training
6. **Use class weights** if labels are imbalanced
7. **Freeze LLM** to keep memory requirements reasonable
8. **Enable AMP** on CUDA for faster training

## Performance Tips

- **Batch size**: 32-64 for cached features, 8-16 for raw frames
- **Learning rate**: Start with 1e-3, decrease if unstable
- **Gradient clipping**: Keep at 1.0 to prevent exploding gradients
- **Epochs**: 10-20 usually sufficient for convergence
- **Workers**: Use `num_workers=4` on multi-core machines

## Troubleshooting

See README.md for detailed troubleshooting guide.

Quick checks:
1. Run `python test_sim.py` to verify setup
2. Check `outputs/*/metrics/*.jsonl` for training curves
3. Verify data normalization
4. Reduce batch size if OOM
5. Check label distributions for class imbalance
