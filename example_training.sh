#!/bin/bash
# Example: Train SIM v0 with default configuration

# Activate virtual environment
source venv/bin/activate

# Run training
python train_sim_v0.py --config configs/sim_v0.yaml

# After training completes, you can run inference:
# python -m sim.inference --checkpoint outputs/sim_v0_run/checkpoints/best.pt --device cpu

echo "Training complete! Check outputs/sim_v0_run/ for results."
