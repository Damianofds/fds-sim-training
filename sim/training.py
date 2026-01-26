"""
Training and evaluation loops for SIM.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from tqdm import tqdm

from sim.metrics import compute_metrics


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    risk_weight: float = 1.0,
    reason_weight: float = 0.5,
    action_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Compute training loss.

    Loss = risk_weight * CE(risk) + reason_weight * CE(reason) + action_weight * CE(action)

    Args:
        outputs: Model outputs
        batch: Input batch with labels
        risk_weight: Weight for risk loss
        reason_weight: Weight for reason loss
        action_weight: Weight for action loss

    Returns:
        Dictionary with total loss and individual losses
    """
    device = outputs["risk_logits"].device

    risk_targets = batch["risk_id"].to(device)
    reason_targets = batch["reason_id"].to(device)
    action_targets = batch["action_id"].to(device)

    # Cross entropy losses
    ce_loss = nn.CrossEntropyLoss()

    risk_loss = ce_loss(outputs["risk_logits"], risk_targets)
    reason_loss = ce_loss(outputs["reason_logits"], reason_targets)
    action_loss = ce_loss(outputs["action_logits"], action_targets)

    # Weighted combination
    total_loss = (
        risk_weight * risk_loss +
        reason_weight * reason_loss +
        action_weight * action_loss
    )

    return {
        "loss": total_loss,
        "risk_loss": risk_loss,
        "reason_loss": reason_loss,
        "action_loss": action_loss,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    grad_clip: float = 1.0,
    use_amp: bool = False,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: PyTorch model
        loader: Training data loader
        optimizer: Optimizer
        scaler: AMP gradient scaler (if using AMP)
        device: Device to train on
        grad_clip: Gradient clipping value
        use_amp: Whether to use automatic mixed precision

    Returns:
        Dictionary of average training metrics
    """
    model.train()

    running_losses = {
        "loss": 0.0,
        "risk_loss": 0.0,
        "reason_loss": 0.0,
        "action_loss": 0.0,
    }

    # For metrics
    all_risk_preds = []
    all_risk_targets = []
    all_reason_preds = []
    all_reason_targets = []
    all_action_preds = []
    all_action_targets = []

    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        optimizer.zero_grad()

        # Forward pass with AMP if enabled
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(batch)
                losses = compute_loss(outputs, batch)
                loss = losses["loss"]

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(batch)
            losses = compute_loss(outputs, batch)
            loss = losses["loss"]

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # Accumulate losses
        for key in running_losses:
            running_losses[key] += losses[key].item()

        # Collect predictions for metrics
        risk_preds = outputs["risk_logits"].argmax(dim=1).cpu().tolist()
        reason_preds = outputs["reason_logits"].argmax(dim=1).cpu().tolist()
        action_preds = outputs["action_logits"].argmax(dim=1).cpu().tolist()

        all_risk_preds.extend(risk_preds)
        all_risk_targets.extend(batch["risk_id"].cpu().tolist())
        all_reason_preds.extend(reason_preds)
        all_reason_targets.extend(batch["reason_id"].cpu().tolist())
        all_action_preds.extend(action_preds)
        all_action_targets.extend(batch["action_id"].cpu().tolist())

        # Update progress bar
        pbar.set_postfix({"loss": loss.item()})

    # Average losses
    num_batches = len(loader)
    avg_losses = {key: value / num_batches for key, value in running_losses.items()}

    # Compute metrics
    metrics = compute_metrics(
        all_risk_preds, all_risk_targets,
        all_reason_preds, all_reason_targets,
        all_action_preds, all_action_targets,
    )

    # Combine losses and metrics
    return {**avg_losses, **metrics}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate model on validation/test set.

    Args:
        model: PyTorch model
        loader: Validation data loader
        device: Device to evaluate on

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    running_losses = {
        "loss": 0.0,
        "risk_loss": 0.0,
        "reason_loss": 0.0,
        "action_loss": 0.0,
    }

    all_risk_preds = []
    all_risk_targets = []
    all_reason_preds = []
    all_reason_targets = []
    all_action_preds = []
    all_action_targets = []

    pbar = tqdm(loader, desc="Evaluating")
    for batch in pbar:
        outputs = model(batch)
        losses = compute_loss(outputs, batch)

        # Accumulate losses
        for key in running_losses:
            running_losses[key] += losses[key].item()

        # Collect predictions
        risk_preds = outputs["risk_logits"].argmax(dim=1).cpu().tolist()
        reason_preds = outputs["reason_logits"].argmax(dim=1).cpu().tolist()
        action_preds = outputs["action_logits"].argmax(dim=1).cpu().tolist()

        all_risk_preds.extend(risk_preds)
        all_risk_targets.extend(batch["risk_id"].cpu().tolist())
        all_reason_preds.extend(reason_preds)
        all_reason_targets.extend(batch["reason_id"].cpu().tolist())
        all_action_preds.extend(action_preds)
        all_action_targets.extend(batch["action_id"].cpu().tolist())

    # Average losses
    num_batches = len(loader)
    avg_losses = {key: value / num_batches for key, value in running_losses.items()}

    # Compute metrics
    metrics = compute_metrics(
        all_risk_preds, all_risk_targets,
        all_reason_preds, all_reason_targets,
        all_action_preds, all_action_targets,
    )

    return {**avg_losses, **metrics}
