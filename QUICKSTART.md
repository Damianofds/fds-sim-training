# Quick Start Guide

Get started with SIM v0 training in 5 minutes.

## 1. Setup (Already Done!)

The virtual environment is already set up with all dependencies installed.

```bash
source venv/bin/activate  # Activate environment
```

## 2. Run Tests

Verify everything works:

```bash
python test_sim.py
```

Expected output: `✓ All tests passed!`

## 3. Train Your First Model

Train with the default configuration (uses dummy data):

```bash
python train_sim_v0.py --config configs/sim_v0.yaml
```

This will:
- Create a GPT-2 based model with frozen backbone
- Train on randomly generated data for 10 epochs
- Save checkpoints to `outputs/sim_v0_run/`
- Log metrics every epoch

Training should take 2-3 minutes on CPU.

## 4. Check Results

After training, explore the outputs:

```bash
# View training metrics
cat outputs/sim_v0_run/metrics/train_metrics.jsonl

# View validation metrics
cat outputs/sim_v0_run/metrics/val_metrics.jsonl

# List checkpoints
ls -lh outputs/sim_v0_run/checkpoints/
```

## 5. Run Inference

Load the trained model and make predictions:

```bash
python -m sim.inference --checkpoint outputs/sim_v0_run/checkpoints/best.pt --device cpu
```

## Next Steps

### Use Your Own Data

1. Create a custom dataset class (see `sim/dataset.py` for examples)
2. Update `configs/sim_v0.yaml` with your dataset path
3. Train with your data

### Customize Training

Edit `configs/sim_v0.yaml` to:
- Change the LLM backbone (`llm_name_or_path`)
- Adjust learning rate and batch size
- Modify architecture parameters (F, K, dv, d)
- Enable mixed precision training (`amp: true`)

### Monitor Training

Track metrics in real-time:

```bash
# In another terminal:
tensorboard --logdir outputs/sim_v0_run/
```

## Common Issues

**Q: Out of memory error**
- Reduce batch_size in config (try 16 or 8)
- Use a smaller LLM (gpt2 is the smallest)

**Q: Training is slow**
- Enable AMP if using CUDA: `amp: true`
- Use cached features: `use_cached_vis_feats: true`
- Increase batch_size if memory allows

**Q: Poor accuracy**
- Train for more epochs (default is 10)
- Check that your data labels are correct
- Verify telemetry is normalized (mean=0, std=1)

## Architecture Overview

```
Input:
  Video (8 frames × 768D features)
  Telemetry (8D: ax,ay,az,gx,gy,gz,v,w)
    ↓
  Vision Projector (trainable)
  Telemetry MLP (trainable)
    ↓
  Temporal Pooling (8 → 4 tokens)
    ↓
  Frozen GPT-2 LLM
    ↓
  Classification Heads (trainable)
    ↓
Output:
  Risk: {none, low, medium, high}
  Reason: {human, obstacle, tight_space, unknown}
  Action: {continue, slow, stop, reverse, turn_left, turn_right}
```

Only the projectors and classification heads are trained (~1.5M params).
The LLM backbone stays frozen (~124M params).

## Resources

- Full documentation: `README.md`
- Dataset interface: `sim/dataset.py`
- Model architecture: `sim/model.py`
- Training config: `configs/sim_v0.yaml`

## Support

If you encounter issues:
1. Check the README for detailed troubleshooting
2. Run `python test_sim.py` to verify setup
3. Review the logs in `outputs/sim_v0_run/logs/`

Happy training! 🚀
