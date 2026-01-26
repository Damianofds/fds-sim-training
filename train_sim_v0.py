#!/usr/bin/env python3
"""
Main training script for SIM v0.

Usage:
    python train_sim_v0.py --config configs/sim_v0.yaml
"""

import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from sim.model import SimV0Model
from sim.dataset import DummySimDataset, collate_fn
from sim.training import train_one_epoch, evaluate
from sim.metrics import print_metrics
from sim.utils import (
    seed_everything,
    get_git_hash,
    save_checkpoint,
    save_config,
    setup_output_dir,
    save_label_maps,
    count_parameters,
    load_config,
    MetricsLogger,
)


def load_dataset(dataset_class_path: str, split_path: str, use_cached_vis_feats: bool):
    """
    Dynamically load dataset class.

    Args:
        dataset_class_path: Import path (e.g., "my_module.MyDataset")
        split_path: Path to data split
        use_cached_vis_feats: Whether to use cached features

    Returns:
        Dataset instance
    """
    # For now, use DummySimDataset if path is "dummy"
    if dataset_class_path == "sim.dataset.DummySimDataset":
        return DummySimDataset(
            num_samples=500,
            use_cached_vis_feats=use_cached_vis_feats,
        )

    # Otherwise, try to import dynamically
    try:
        module_path, class_name = dataset_class_path.rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        dataset_class = getattr(module, class_name)
        return dataset_class(split_path, use_cached_vis_feats=use_cached_vis_feats)
    except Exception as e:
        print(f"Error loading dataset class {dataset_class_path}: {e}")
        print("Falling back to DummySimDataset")
        return DummySimDataset(
            num_samples=500,
            use_cached_vis_feats=use_cached_vis_feats,
        )


def main():
    parser = argparse.ArgumentParser(description="Train SIM v0")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    print(f"Loaded config from {config_path}")

    # Extract config sections
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    output_config = config["output"]

    # Setup
    seed = training_config.get("seed", 42)
    seed_everything(seed)
    print(f"Set random seed: {seed}")

    # Git hash
    git_hash = get_git_hash()
    if git_hash:
        print(f"Git commit: {git_hash}")

    # Output directory
    output_dir = Path(output_config["output_dir"])
    setup_output_dir(output_dir)

    # Save resolved config
    save_config(config, output_dir)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = load_dataset(
        data_config["dataset_class_path"],
        data_config["train_split_path"],
        model_config["use_cached_vis_feats"],
    )
    val_dataset = load_dataset(
        data_config["dataset_class_path"],
        data_config["val_split_path"],
        model_config["use_cached_vis_feats"],
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=data_config.get("num_workers", 0),
        pin_memory=True if device.type == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=data_config.get("num_workers", 0),
        pin_memory=True if device.type == "cuda" else False,
    )

    # Create model
    print("\nInitializing model...")
    model = SimV0Model(
        llm_name_or_path=model_config["llm_name_or_path"],
        dv=model_config["dv"],
        d=model_config["d"],
        F=model_config["F"],
        K=model_config["K"],
        use_cached_vis_feats=model_config["use_cached_vis_feats"],
        vision_encoder_name=model_config.get("vision_encoder_name"),
    )
    model = model.to(device)

    # Verify trainable parameters
    trainable, total = count_parameters(model)
    assert trainable > 0, "No trainable parameters!"
    print(f"\nParameter check: {trainable:,} trainable / {total:,} total")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["lr"],
        weight_decay=training_config.get("weight_decay", 0.01),
    )

    # AMP scaler
    use_amp = training_config.get("amp", False) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("Using automatic mixed precision (AMP)")

    # Metrics loggers
    train_logger = MetricsLogger(output_dir / "metrics" / "train_metrics.jsonl")
    val_logger = MetricsLogger(output_dir / "metrics" / "val_metrics.jsonl")

    # Training loop
    print("\nStarting training...")
    print("=" * 60)

    best_val_metric = 0.0
    best_epoch = 0

    for epoch in range(1, training_config["epochs"] + 1):
        print(f"\nEpoch {epoch}/{training_config['epochs']}")
        print("-" * 60)

        # Train
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_clip=training_config.get("grad_clip", 1.0),
            use_amp=use_amp,
        )

        print_metrics(train_metrics, prefix="Train ")
        train_logger.log({"epoch": epoch, **train_metrics})

        # Validate
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
        )

        print_metrics(val_metrics, prefix="Val ")
        val_logger.log({"epoch": epoch, **val_metrics})

        # Save last checkpoint
        save_checkpoint(
            output_dir=output_dir,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=val_metrics,
            metadata=model.get_metadata(),
            checkpoint_name="last.pt",
        )

        # Save best checkpoint (based on risk_high_recall)
        val_score = val_metrics.get("risk_high_recall", 0.0)
        if val_score > best_val_metric:
            best_val_metric = val_score
            best_epoch = epoch
            save_checkpoint(
                output_dir=output_dir,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                metadata=model.get_metadata(),
                checkpoint_name="best.pt",
            )
            print(f"New best model! risk_high_recall: {best_val_metric:.4f}")

        # Optionally save epoch checkpoint
        if training_config.get("save_every_epoch", False):
            save_checkpoint(
                output_dir=output_dir,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                metadata=model.get_metadata(),
                checkpoint_name=f"epoch_{epoch:03d}.pt",
            )

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best epoch: {best_epoch}")
    print(f"Best risk_high_recall: {best_val_metric:.4f}")
    print(f"Checkpoints saved to: {output_dir / 'checkpoints'}")

    # Save label maps
    save_label_maps(model.get_metadata()["label_maps"], output_dir)

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
