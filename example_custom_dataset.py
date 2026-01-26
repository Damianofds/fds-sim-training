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
import shutil
import argparse
import cv2

from sim.dataset import SimDataset
import torch.nn.functional as F
from torchvision.io import read_video


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

    def __init__(
        self,
        data_path: str,
        use_cached_vis_feats: bool = False,
        num_frames: int = 8,
        frame_size=(224, 224),
        normalize_telemetry: bool = True,
    ):
        """
        Load raw frames for each sample.

                Directory convention (only supported):
                - Per-sample directory: {data_root}/{sample_id}/training.mp4 and
                    {data_root}/{sample_id}/telemetry.npy

                Videos are read with `torchvision.io.read_video`, uniformly sampled or
                padded to `num_frames`, resized to `frame_size`, and returned as a
                float tensor in range [0,1] with shape [F, 3, H, W].
        """
        super().__init__(data_path, use_cached_vis_feats)

        self.data_root = Path(data_path)
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.normalize_telemetry = normalize_telemetry

        # Load samples metadata (reuse same samples.pkl format)
        samples_file = self.data_root / "samples.pkl"
        if not samples_file.exists():
            raise FileNotFoundError(f"Samples file not found: {samples_file}")

        with open(samples_file, "rb") as f:
            self.samples = pickle.load(f)

        # Compute telemetry normalization stats (dummy by default)
        if self.normalize_telemetry:
            self._compute_telemetry_stats()

    def _compute_telemetry_stats(self):
        """Compute mean/std for telemetry normalization."""
        self.telemetry_mean = np.zeros(8)
        self.telemetry_std = np.ones(8)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_meta = self.samples[idx]
        sample_id = sample_meta.get("id", str(idx))

        # Resolve video path (only per-sample layout supported)
        per_sample_video = self.data_root / sample_id / "training.mp4"
        if not per_sample_video.exists():
            raise FileNotFoundError(f"Video not found for sample {sample_id}: {per_sample_video}")
        video_path = str(per_sample_video)

        # Read video: returns tensor [T, H, W, C] uint8
        video, _, _ = read_video(video_path, pts_unit="sec")
        if video.numel() == 0:
            raise RuntimeError(f"Empty video for sample {sample_id}: {video_path}")

        # Convert to float [T, C, H, W] in [0,1]
        video = video.permute(0, 3, 1, 2).float() / 255.0

        T = video.shape[0]
        if T >= self.num_frames:
            indices = torch.linspace(0, T - 1, steps=self.num_frames).long()
            frames = video[indices]
        else:
            # pad by repeating last frame
            pad = self.num_frames - T
            last = video[-1:].repeat(pad, 1, 1, 1)
            frames = torch.cat([video, last], dim=0)

        # Resize to target frame_size if necessary
        H, W = self.frame_size
        if frames.shape[2] != H or frames.shape[3] != W:
            frames = F.interpolate(frames, size=(H, W), mode="bilinear", align_corners=False)

        # Load telemetry (only per-sample layout supported)
        per_sample_tel = self.data_root / sample_id / "telemetry.npy"
        if not per_sample_tel.exists():
            raise FileNotFoundError(f"Telemetry not found for sample {sample_id}: {per_sample_tel}")
        tel_path = per_sample_tel

        telemetry = np.load(tel_path)
        if self.normalize_telemetry:
            telemetry = (telemetry - self.telemetry_mean) / self.telemetry_std

        telemetry = torch.from_numpy(telemetry).float()

        out = {
            "frames": frames,  # [F, 3, H, W]
            "telemetry": telemetry,
            "risk_id": sample_meta.get("risk_label", 0),
            "reason_id": sample_meta.get("reason_label", 0),
            "action_id": sample_meta.get("action_label", 0),
            "sample_id": sample_id,
            "weight": sample_meta.get("weight", 1.0),
        }

        return out


def create_dummy_dataset_files(output_dir: str = "/tmp/data/example"):
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


def make_dummy_videos(data_dir: str = "/tmp/data/train", num_frames: int = 8, frame_size=(224, 224), fps: int = 5, overwrite: bool = False):
    """
    Create per-sample directories with `training.mp4` and copy `telemetry.npy`.

    Args:
        data_dir: root dataset directory containing `samples.pkl` and `telemetry/`
        num_frames: number of frames to write into each `training.mp4`
        frame_size: (H, W) target frame size used by the dataset; writer needs (W,H)
        fps: frames per second for the video file
        overwrite: if True, overwrite existing training.mp4 and telemetry.npy
    """
    data_root = Path(data_dir)
    samples_file = data_root / "samples.pkl"
    if not samples_file.exists():
        raise FileNotFoundError(f"samples.pkl not found in {data_root}")

    with open(samples_file, "rb") as f:
        samples = pickle.load(f)

    tel_dir = data_root / "telemetry"
    H, W = frame_size
    width, height = W, H

    for meta in samples:
        sid = meta["id"]
        sample_dir = data_root / sid
        sample_dir.mkdir(parents=True, exist_ok=True)

        # copy telemetry into per-sample dir
        src_tel = tel_dir / f"{sid}.npy"
        dst_tel = sample_dir / "telemetry.npy"
        if src_tel.exists() and (overwrite or not dst_tel.exists()):
            shutil.copy(src_tel, dst_tel)

        # write training.mp4 with random frames
        out_mp4 = sample_dir / "training.mp4"
        if out_mp4.exists() and not overwrite:
            continue

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_mp4), fourcc, float(fps), (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"cv2 cannot open VideoWriter for {out_mp4}")

        for i in range(num_frames):
            base_val = (10 + (i * 20) % 240)
            base = np.full((height, width, 3), fill_value=base_val, dtype=np.uint8)
            noise = np.random.randint(0, 20, (height, width, 3), dtype=np.uint8)
            frame = ((base.astype(np.uint16) + noise.astype(np.uint16)) % 256).astype(np.uint8)
            writer.write(frame)
        writer.release()

    print(f"Wrote training.mp4 + telemetry.npy for {len(samples)} samples in {data_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", "-o", default="/tmp/data/train", help="Output dataset directory to create (default: /tmp/data/train)")
    parser.add_argument("--num-frames", type=int, default=8, help="Number of frames to write into each training.mp4")
    parser.add_argument("--frame-height", type=int, default=224)
    parser.add_argument("--frame-width", type=int, default=224)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing per-sample files")
    args = parser.parse_args()

    # Create samples and generate per-sample videos in one run
    target_dir = args.out
    print("Creating example dataset files...")
    create_dummy_dataset_files(target_dir)

    print("Generating training.mp4 files for each sample...")
    try:
        make_dummy_videos(data_dir=target_dir, num_frames=args.num_frames, frame_size=(args.frame_height, args.frame_width), fps=args.fps, overwrite=args.overwrite)
    except Exception as e:
        print(f"Warning: make_dummy_videos failed: {e}")

    # Load and test dataset
    print("\nTesting dataset loading...")
    dataset = RoverRawFramesDataset(target_dir)
    print(f"Dataset size: {len(dataset)}")

    # Get a sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    # print(f"  vis_feats shape: {sample['vis_feats'].shape}")
    print(f"  telemetry shape: {sample['telemetry'].shape}")
    print(f"  risk_id: {sample['risk_id']} ({dataset.RISK_LABELS[sample['risk_id']]})")
    print(f"  action_id: {sample['action_id']} ({dataset.ACTION_LABELS[sample['action_id']]})")

    print("\n✓ Custom dataset works correctly!")
    print("\nNext steps:")
    print("1. Modify RoverVideoDataset to load your actual data")
    print("2. Update configs/sim_v0.yaml with your dataset class path")
    print("3. Run: python train_sim_v0.py --config configs/sim_v0.yaml")
