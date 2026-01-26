#!/usr/bin/env python3
"""
Quick test script to validate SIM v0 implementation.

Tests:
1. Model initialization
2. Forward pass
3. Training loop (1 iteration)
4. Metrics computation
5. Checkpoint save/load
"""

import torch
from torch.utils.data import DataLoader

from sim.model import SimV0Model
from sim.dataset import DummySimDataset, collate_fn
from sim.training import train_one_epoch, evaluate
from sim.metrics import compute_metrics
from sim.utils import seed_everything, save_checkpoint, load_checkpoint
from pathlib import Path
import tempfile


def test_model_initialization():
    """Test model creation."""
    print("\n" + "=" * 60)
    print("Test 1: Model Initialization")
    print("=" * 60)

    model = SimV0Model(
        llm_name_or_path="gpt2",
        dv=768,
        d=768,
        F=8,
        K=4,
        use_cached_vis_feats=True,
    )

    # Check trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"✓ Model created successfully")
    print(f"  Trainable params: {trainable:,}")
    print(f"  Total params: {total:,}")
    print(f"  Frozen params: {total - trainable:,}")

    assert trainable > 0, "No trainable parameters!"
    assert trainable < total, "All parameters trainable (LLM not frozen!)"

    return model


def test_forward_pass(model):
    """Test forward pass."""
    print("\n" + "=" * 60)
    print("Test 2: Forward Pass")
    print("=" * 60)

    batch_size = 4
    batch = {
        "vis_feats": torch.randn(batch_size, 8, 768),
        "telemetry": torch.randn(batch_size, 8),
        "risk_id": torch.randint(0, 4, (batch_size,)),
        "reason_id": torch.randint(0, 4, (batch_size,)),
        "action_id": torch.randint(0, 6, (batch_size,)),
    }

    outputs = model(batch)

    print(f"✓ Forward pass successful")
    print(f"  Batch size: {batch_size}")
    print(f"  Risk logits shape: {outputs['risk_logits'].shape}")
    print(f"  Reason logits shape: {outputs['reason_logits'].shape}")
    print(f"  Action logits shape: {outputs['action_logits'].shape}")

    assert outputs["risk_logits"].shape == (batch_size, 4)
    assert outputs["reason_logits"].shape == (batch_size, 4)
    assert outputs["action_logits"].shape == (batch_size, 6)

    return outputs


def test_prediction(model):
    """Test prediction function."""
    print("\n" + "=" * 60)
    print("Test 3: Prediction")
    print("=" * 60)

    batch = {
        "vis_feats": torch.randn(2, 8, 768),
        "telemetry": torch.randn(2, 8),
        "sample_id": ["sample_1", "sample_2"],
    }

    predictions = model.predict_json(batch)

    print(f"✓ Predictions generated successfully")
    print(f"  Number of predictions: {len(predictions)}")
    print(f"  Sample prediction:")
    for key, value in predictions[0].items():
        print(f"    {key}: {value}")

    assert len(predictions) == 2
    assert "risk" in predictions[0]
    assert "action" in predictions[0]

    return predictions


def test_training_loop(model):
    """Test training for 1 iteration."""
    print("\n" + "=" * 60)
    print("Test 4: Training Loop")
    print("=" * 60)

    seed_everything(42)

    # Create small dataset
    dataset = DummySimDataset(num_samples=16, use_cached_vis_feats=True)
    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    device = torch.device("cpu")

    # Train for 1 epoch
    metrics = train_one_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        scaler=None,
        device=device,
        grad_clip=1.0,
        use_amp=False,
    )

    print(f"✓ Training completed successfully")
    print(f"  Train loss: {metrics['loss']:.4f}")
    print(f"  Risk accuracy: {metrics['risk_accuracy']:.4f}")
    print(f"  Action accuracy: {metrics['action_accuracy']:.4f}")

    assert "loss" in metrics
    assert metrics["loss"] > 0

    return metrics


def test_evaluation(model):
    """Test evaluation."""
    print("\n" + "=" * 60)
    print("Test 5: Evaluation")
    print("=" * 60)

    dataset = DummySimDataset(num_samples=16, use_cached_vis_feats=True)
    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)

    device = torch.device("cpu")

    metrics = evaluate(model=model, loader=loader, device=device)

    print(f"✓ Evaluation completed successfully")
    print(f"  Val loss: {metrics['loss']:.4f}")
    print(f"  Risk high recall: {metrics['risk_high_recall']:.4f}")
    print(f"  Risk high FN rate: {metrics['risk_high_fn_rate']:.4f}")

    assert "risk_high_recall" in metrics
    assert "macro_f1" in metrics

    return metrics


def test_checkpointing(model):
    """Test checkpoint save/load."""
    print("\n" + "=" * 60)
    print("Test 6: Checkpointing")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        metrics = {"loss": 0.5, "risk_high_recall": 0.8}
        metadata = model.get_metadata()

        # Save checkpoint
        save_checkpoint(
            output_dir=output_dir,
            model=model,
            optimizer=optimizer,
            epoch=1,
            metrics=metrics,
            metadata=metadata,
            checkpoint_name="test.pt",
        )

        checkpoint_path = output_dir / "checkpoints" / "test.pt"
        assert checkpoint_path.exists(), "Checkpoint not saved!"

        # Create new model and load
        model2 = SimV0Model(
            llm_name_or_path="gpt2",
            dv=768,
            d=768,
            F=8,
            K=4,
            use_cached_vis_feats=True,
        )

        info = load_checkpoint(checkpoint_path, model2, optimizer)

        print(f"✓ Checkpoint saved and loaded successfully")
        print(f"  Checkpoint path: {checkpoint_path}")
        print(f"  Epoch: {info['epoch']}")
        print(f"  Metrics: {info['metrics']}")

        assert info["epoch"] == 1
        assert info["metrics"]["loss"] == 0.5


def test_metrics_computation():
    """Test metrics computation."""
    print("\n" + "=" * 60)
    print("Test 7: Metrics Computation")
    print("=" * 60)

    # Create dummy predictions and targets
    risk_preds = [0, 1, 2, 3, 3, 3, 2, 1]
    risk_targets = [0, 1, 2, 3, 3, 2, 2, 1]
    reason_preds = [0, 1, 2, 3, 0, 1, 2, 3]
    reason_targets = [0, 1, 2, 3, 1, 1, 2, 3]
    action_preds = [0, 1, 2, 3, 4, 5, 0, 1]
    action_targets = [0, 1, 2, 3, 4, 5, 1, 1]

    metrics = compute_metrics(
        risk_preds, risk_targets,
        reason_preds, reason_targets,
        action_preds, action_targets,
    )

    print(f"✓ Metrics computed successfully")
    for key, value in sorted(metrics.items()):
        print(f"  {key}: {value:.4f}")

    assert "risk_high_recall" in metrics
    assert "action_accuracy" in metrics
    assert "macro_f1" in metrics


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("SIM v0 Implementation Tests")
    print("=" * 60)

    try:
        # Test 1: Model initialization
        model = test_model_initialization()

        # Test 2: Forward pass
        test_forward_pass(model)

        # Test 3: Prediction
        test_prediction(model)

        # Test 4: Training loop
        test_training_loop(model)

        # Test 5: Evaluation
        test_evaluation(model)

        # Test 6: Checkpointing
        test_checkpointing(model)

        # Test 7: Metrics
        test_metrics_computation()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ Test failed with error: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    main()
