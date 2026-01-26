"""
Utility functions for training: seeding, checkpointing, config management.
"""

import os
import random
import torch
import numpy as np
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess


def seed_everything(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make cudnn deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_git_hash() -> Optional[str]:
    """
    Get current git commit hash if available.

    Returns:
        Git hash string or None if not in a git repo
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def save_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    metadata: Dict[str, Any],
    checkpoint_name: str = "checkpoint.pt"
):
    """
    Save model checkpoint with optimizer state and metadata.

    Args:
        output_dir: Directory to save checkpoint
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch number
        metrics: Dictionary of metrics
        metadata: Model metadata (architecture params, label maps, etc.)
        checkpoint_name: Name of checkpoint file
    """
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "metadata": metadata,
    }

    checkpoint_path = checkpoint_dir / checkpoint_name
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")


def load_checkpoint(checkpoint_path: Path, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model to load state into
        optimizer: Optional optimizer to load state into

    Returns:
        Dictionary with epoch, metrics, and metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return {
        "epoch": checkpoint["epoch"],
        "metrics": checkpoint.get("metrics", {}),
        "metadata": checkpoint.get("metadata", {}),
    }


def save_config(config: Dict[str, Any], output_dir: Path):
    """
    Save resolved configuration to output directory.

    Args:
        config: Configuration dictionary
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config_resolved.yaml"

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Saved config: {config_path}")


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config YAML file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_output_dir(output_dir: Path):
    """
    Create output directory structure.

    Args:
        output_dir: Root output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "metrics").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "export").mkdir(exist_ok=True)

    print(f"Output directory: {output_dir}")


def save_label_maps(label_maps: Dict[str, Dict[int, str]], output_dir: Path):
    """
    Save label ID to name mappings.

    Args:
        label_maps: Dictionary of label mappings (e.g., risk_map, action_map)
        output_dir: Output directory
    """
    export_dir = output_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    labels_path = export_dir / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(label_maps, f, indent=2)

    print(f"Saved label maps: {labels_path}")


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """
    Count trainable and total parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (trainable_params, total_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


class MetricsLogger:
    """Simple JSONL metrics logger."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: Dict[str, Any]):
        """Append metrics as a JSON line."""
        with open(self.log_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")
