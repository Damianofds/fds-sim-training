"""
Metrics computation for SIM training.

Key metrics:
- risk_high_recall: recall for class "high" (class 3)
- risk_high_fn_rate: false negative rate on "high"
- action_accuracy: overall accuracy
- macro_f1: macro-averaged F1 for risk and action
"""

import torch
import numpy as np
from typing import Dict, List
from sklearn.metrics import recall_score, f1_score, accuracy_score, confusion_matrix


def compute_metrics(
    risk_preds: List[int],
    risk_targets: List[int],
    reason_preds: List[int],
    reason_targets: List[int],
    action_preds: List[int],
    action_targets: List[int],
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        risk_preds: List of predicted risk class IDs
        risk_targets: List of target risk class IDs
        reason_preds: List of predicted reason class IDs
        reason_targets: List of target reason class IDs
        action_preds: List of predicted action class IDs
        action_targets: List of target action class IDs

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Convert to numpy arrays
    risk_preds = np.array(risk_preds)
    risk_targets = np.array(risk_targets)
    reason_preds = np.array(reason_preds)
    reason_targets = np.array(reason_targets)
    action_preds = np.array(action_preds)
    action_targets = np.array(action_targets)

    # Risk metrics
    risk_acc = accuracy_score(risk_targets, risk_preds)
    metrics["risk_accuracy"] = float(risk_acc)

    # Risk high recall (class 3)
    high_risk_class = 3
    risk_recall_per_class = recall_score(
        risk_targets, risk_preds, labels=[0, 1, 2, 3], average=None, zero_division=0
    )
    if len(risk_recall_per_class) > high_risk_class:
        metrics["risk_high_recall"] = float(risk_recall_per_class[high_risk_class])
    else:
        metrics["risk_high_recall"] = 0.0

    # Risk high false negative rate
    # FN rate = 1 - recall
    metrics["risk_high_fn_rate"] = 1.0 - metrics["risk_high_recall"]

    # Risk F1 (macro)
    risk_f1 = f1_score(risk_targets, risk_preds, average="macro", zero_division=0)
    metrics["risk_f1_macro"] = float(risk_f1)

    # Reason metrics
    reason_acc = accuracy_score(reason_targets, reason_preds)
    metrics["reason_accuracy"] = float(reason_acc)

    reason_f1 = f1_score(reason_targets, reason_preds, average="macro", zero_division=0)
    metrics["reason_f1_macro"] = float(reason_f1)

    # Action metrics
    action_acc = accuracy_score(action_targets, action_preds)
    metrics["action_accuracy"] = float(action_acc)

    action_f1 = f1_score(action_targets, action_preds, average="macro", zero_division=0)
    metrics["action_f1_macro"] = float(action_f1)

    # Overall macro F1 (risk + action)
    metrics["macro_f1"] = (risk_f1 + action_f1) / 2.0

    return metrics


def compute_confusion_matrix(targets: List[int], preds: List[int], num_classes: int) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        targets: Ground truth labels
        preds: Predicted labels
        num_classes: Number of classes

    Returns:
        Confusion matrix as numpy array
    """
    return confusion_matrix(targets, preds, labels=list(range(num_classes)))


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    Pretty print metrics.

    Args:
        metrics: Dictionary of metrics
        prefix: Optional prefix for metric names (e.g., "train_" or "val_")
    """
    print(f"\n{prefix}Metrics:")
    print("-" * 50)
    for key, value in sorted(metrics.items()):
        print(f"  {key:25s}: {value:.4f}")
    print("-" * 50)
