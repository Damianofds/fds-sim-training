"""
Base dataset interface for SIM training.

Users should inherit from SimDataset and implement their own data loading logic.
"""

from torch.utils.data import Dataset
from typing import Dict, Any
import torch


class SimDataset(Dataset):
    """
    Abstract base class for SIM datasets.

    Subclasses must implement __len__ and __getitem__.

    Expected output format from __getitem__:
        {
            # Video representation (choose ONE):
            "frames": torch.Tensor [F, 3, H, W] (if using raw frames) OR
            "vis_feats": torch.Tensor [F, dv] (if using cached features),

            # Telemetry:
            "telemetry": torch.Tensor [8],  # [ax, ay, az, gx, gy, gz, v, w]

            # Labels:
            "risk_id": int in {0,1,2,3},      # none,low,med,high
            "reason_id": int in {0,1,2,3},    # human,obstacle,tight_space,unknown
            "action_id": int in {0..5},       # continue,slow,stop,reverse,turn_left,turn_right

            # Optional:
            "weight": float (for class weighting),
            "sample_id": str (for debugging),
        }
    """

    # Label mappings (subclasses can override)
    RISK_LABELS = {
        0: "none",
        1: "low",
        2: "medium",
        3: "high",
    }

    REASON_LABELS = {
        0: "human",
        1: "obstacle",
        2: "tight_space",
        3: "unknown",
    }

    ACTION_LABELS = {
        0: "continue",
        1: "slow",
        2: "stop",
        3: "reverse",
        4: "turn_left",
        5: "turn_right",
    }

    def __init__(self, data_path: str, use_cached_vis_feats: bool = True):
        """
        Initialize dataset.

        Args:
            data_path: Path to dataset
            use_cached_vis_feats: Whether to use cached vision features or raw frames
        """
        self.data_path = data_path
        self.use_cached_vis_feats = use_cached_vis_feats

    def __len__(self) -> int:
        """Return dataset size."""
        raise NotImplementedError("Subclasses must implement __len__")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with keys as described in class docstring
        """
        raise NotImplementedError("Subclasses must implement __getitem__")


class DummySimDataset(SimDataset):
    """
    Dummy dataset for testing purposes.

    Generates random data that matches the expected format.
    """

    def __init__(
        self,
        num_samples: int = 100,
        use_cached_vis_feats: bool = True,
        F: int = 8,
        dv: int = 768,
        H: int = 224,
        W: int = 224,
    ):
        """
        Initialize dummy dataset.

        Args:
            num_samples: Number of samples to generate
            use_cached_vis_feats: Whether to use cached features or raw frames
            F: Number of frames
            dv: Vision feature dimension
            H: Frame height (if using raw frames)
            W: Frame width (if using raw frames)
        """
        super().__init__("dummy", use_cached_vis_feats)
        self.num_samples = num_samples
        self.F = F
        self.dv = dv
        self.H = H
        self.W = W

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = {
            "telemetry": torch.randn(8),  # [ax, ay, az, gx, gy, gz, v, w]
            "risk_id": torch.randint(0, 4, (1,)).item(),
            "reason_id": torch.randint(0, 4, (1,)).item(),
            "action_id": torch.randint(0, 6, (1,)).item(),
            "weight": 1.0,
            "sample_id": f"dummy_{idx}",
        }

        if self.use_cached_vis_feats:
            sample["vis_feats"] = torch.randn(self.F, self.dv)
        else:
            sample["frames"] = torch.randn(self.F, 3, self.H, self.W)

        return sample


def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.

    Args:
        batch: List of samples from dataset

    Returns:
        Batched dictionary
    """
    # Stack tensors
    collated = {}

    # Handle vision input
    if "vis_feats" in batch[0]:
        collated["vis_feats"] = torch.stack([item["vis_feats"] for item in batch])
    elif "frames" in batch[0]:
        collated["frames"] = torch.stack([item["frames"] for item in batch])

    # Stack telemetry
    collated["telemetry"] = torch.stack([item["telemetry"] for item in batch])

    # Stack labels (as long tensors)
    collated["risk_id"] = torch.tensor([item["risk_id"] for item in batch], dtype=torch.long)
    collated["reason_id"] = torch.tensor([item["reason_id"] for item in batch], dtype=torch.long)
    collated["action_id"] = torch.tensor([item["action_id"] for item in batch], dtype=torch.long)

    # Optional: weights
    if "weight" in batch[0]:
        collated["weight"] = torch.tensor([item["weight"] for item in batch], dtype=torch.float32)

    # Optional: sample IDs (keep as list)
    if "sample_id" in batch[0]:
        collated["sample_id"] = [item["sample_id"] for item in batch]

    return collated
