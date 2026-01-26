"""
Example: Custom dataset implementation for SIM v0.

This shows how to create your own dataset class for training.
Modify this template to load your actual rover data.
"""

import torch
import numpy as np
from pathlib import Path
import pickle
from typing import Dict, Any

from sim.dataset import SimDataset


class RoverVideoDataset(SimDataset):
    """
    Example dataset that loads rover video clips and telemetry.

    Expected data format:
    - data_path/: Root directory
      - samples.pkl: Pickle file with list of sample metadata
      - features/: Directory with cached vision features (.npy files)
      - telemetry/: Directory with telemetry data (.npy files)

    Each sample in samples.pkl should have:
    {
        "id": str,
        "features_path": str,  # Relative path to .npy file
        "telemetry_path": str, # Relative path to .npy file
        "risk_label": int,     # 0-3
        "reason_label": int,   # 0-3
        "action_label": int,   # 0-5
        "weight": float,       # Optional, for class weighting
    }
    """

    def __init__(
        self,
        data_path: str,
        use_cached_vis_feats: bool = True,
        normalize_telemetry: bool = True,
    ):
        """
        Initialize dataset.

        Args:
            data_path: Path to dataset root
            use_cached_vis_feats: Whether using cached features
            normalize_telemetry: Whether to normalize telemetry
        """
        super().__init__(data_path, use_cached_vis_feats)

        self.data_root = Path(data_path)
        self.normalize_telemetry = normalize_telemetry

        # Load sample metadata
        samples_file = self.data_root / "samples.pkl"
        if not samples_file.exists():
            raise FileNotFoundError(f"Samples file not found: {samples_file}")

        with open(samples_file, "rb") as f:
            self.samples = pickle.load(f)

        print(f"Loaded {len(self.samples)} samples from {data_path}")

        # Compute telemetry normalization statistics if needed
        if self.normalize_telemetry:
            self._compute_telemetry_stats()

    def _compute_telemetry_stats(self):
        """Compute mean/std for telemetry normalization."""
        # In practice, compute this once and save it
        # For now, use dummy values
        self.telemetry_mean = np.zeros(8)
        self.telemetry_std = np.ones(8)

        # Example: compute from a subset of data
        # telemetry_samples = []
        # for i in range(min(1000, len(self.samples))):
        #     telem_path = self.data_root / self.samples[i]["telemetry_path"]
        #     telem = np.load(telem_path)
        #     telemetry_samples.append(telem)
        # telemetry_samples = np.array(telemetry_samples)
        # self.telemetry_mean = telemetry_samples.mean(axis=0)
        # self.telemetry_std = telemetry_samples.std(axis=0) + 1e-8

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        sample_meta = self.samples[idx]

        # Load vision features
        features_path = self.data_root / sample_meta["features_path"]
        vis_feats = np.load(features_path)  # Expected shape: [F, dv]
        vis_feats = torch.from_numpy(vis_feats).float()

        # Load telemetry
        telemetry_path = self.data_root / sample_meta["telemetry_path"]
        telemetry = np.load(telemetry_path)  # Expected shape: [8]

        # Normalize telemetry
        if self.normalize_telemetry:
            telemetry = (telemetry - self.telemetry_mean) / self.telemetry_std

        telemetry = torch.from_numpy(telemetry).float()

        # Create sample dict
        sample = {
            "vis_feats": vis_feats,
            "telemetry": telemetry,
            "risk_id": sample_meta["risk_label"],
            "reason_id": sample_meta["reason_label"],
            "action_id": sample_meta["action_label"],
            "sample_id": sample_meta["id"],
        }

        # Optional: sample weight
        if "weight" in sample_meta:
            sample["weight"] = sample_meta["weight"]
        else:
            sample["weight"] = 1.0

        return sample


class RoverRawFramesDataset(SimDataset):
    """
    Example dataset that loads raw video frames (not cached features).

    Use this if you want the vision encoder to process frames during training.
    Note: This is slower than using cached features.
    """

    def __init__(self, data_path: str, use_cached_vis_feats: bool = False):
        super().__init__(data_path, use_cached_vis_feats)
        # Implementation similar to above, but load frames instead
        # frames = torch.load(frame_path)  # [F, 3, H, W]
        raise NotImplementedError("Implement frame loading logic")

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


def create_dummy_dataset_files(output_dir: str = "data/example"):
    """
    Create dummy dataset files for testing.

    This creates a minimal dataset in the expected format.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    features_dir = output_path / "features"
    telemetry_dir = output_path / "telemetry"
    features_dir.mkdir(exist_ok=True)
    telemetry_dir.mkdir(exist_ok=True)

    # Create dummy samples
    num_samples = 100
    samples = []

    for i in range(num_samples):
        sample_id = f"sample_{i:04d}"

        # Save dummy features
        features = np.random.randn(8, 768).astype(np.float32)
        features_path = features_dir / f"{sample_id}.npy"
        np.save(features_path, features)

        # Save dummy telemetry
        telemetry = np.random.randn(8).astype(np.float32)
        telemetry_path = telemetry_dir / f"{sample_id}.npy"
        np.save(telemetry_path, telemetry)

        # Sample metadata
        sample_meta = {
            "id": sample_id,
            "features_path": f"features/{sample_id}.npy",
            "telemetry_path": f"telemetry/{sample_id}.npy",
            "risk_label": np.random.randint(0, 4),
            "reason_label": np.random.randint(0, 4),
            "action_label": np.random.randint(0, 6),
            "weight": 1.0,
        }
        samples.append(sample_meta)

    # Save samples metadata
    with open(output_path / "samples.pkl", "wb") as f:
        pickle.dump(samples, f)

    print(f"Created {num_samples} dummy samples in {output_path}")
    print(f"Total size: {sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / 1024:.1f} KB")


if __name__ == "__main__":
    # Example 1: Create dummy dataset files
    print("Creating example dataset files...")
    create_dummy_dataset_files("data/train")
    create_dummy_dataset_files("data/val")

    # Example 2: Load and test dataset
    print("\nTesting dataset loading...")
    dataset = RoverVideoDataset("data/train", use_cached_vis_feats=True)
    print(f"Dataset size: {len(dataset)}")

    # Get a sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"  vis_feats shape: {sample['vis_feats'].shape}")
    print(f"  telemetry shape: {sample['telemetry'].shape}")
    print(f"  risk_id: {sample['risk_id']} ({dataset.RISK_LABELS[sample['risk_id']]})")
    print(f"  action_id: {sample['action_id']} ({dataset.ACTION_LABELS[sample['action_id']]})")

    print("\n✓ Custom dataset works correctly!")
    print("\nNext steps:")
    print("1. Modify RoverVideoDataset to load your actual data")
    print("2. Update configs/sim_v0.yaml with your dataset class path")
    print("3. Run: python train_sim_v0.py --config configs/sim_v0.yaml")
