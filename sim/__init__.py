"""
SIM v0 Training Package

Spatial Intent Model with frozen LLM for rover navigation.
"""

__version__ = "0.1.0"

from sim.model import SimV0Model
from sim.dataset import SimDataset
from sim.training import train_one_epoch, evaluate
from sim.metrics import compute_metrics

__all__ = [
    "SimV0Model",
    "SimDataset",
    "train_one_epoch",
    "evaluate",
    "compute_metrics",
]
