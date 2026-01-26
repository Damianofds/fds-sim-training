"""
Minimal inference wrapper for SIM v0.

Load a trained checkpoint and run inference on new data.
"""

import torch
from pathlib import Path
from typing import Dict, List, Any
import json

from sim.model import SimV0Model


class SimV0Inference:
    """
    Inference wrapper for trained SIM v0 models.
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        """
        Initialize inference wrapper.

        Args:
            checkpoint_path: Path to trained checkpoint (.pt file)
            device: Device to run on ("cpu" or "cuda")
        """
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)

        # Load checkpoint
        print(f"Loading checkpoint: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Extract metadata
        self.metadata = checkpoint["metadata"]
        self.label_maps = self.metadata["label_maps"]

        # Create model
        # Note: You need to know the LLM name used during training
        # This should be saved in metadata in a future version
        print("Initializing model...")
        self.model = SimV0Model(
            llm_name_or_path="gpt2",  # Should be saved in metadata
            dv=self.metadata["dv"],
            d=self.metadata["d"],
            F=self.metadata["F"],
            K=self.metadata["K"],
            use_cached_vis_feats=self.metadata["use_cached_vis_feats"],
        )

        # Load weights
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded successfully!")
        print(f"Epoch: {checkpoint['epoch']}")
        print(f"Metrics: {checkpoint['metrics']}")

    @torch.no_grad()
    def predict(self, batch: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
        """
        Run inference on a batch.

        Args:
            batch: Input batch matching training format

        Returns:
            List of prediction dictionaries
        """
        return self.model.predict_json(batch)

    @torch.no_grad()
    def predict_single(
        self,
        vis_feats: torch.Tensor,
        telemetry: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Run inference on a single sample.

        Args:
            vis_feats: Vision features [F, dv]
            telemetry: Telemetry [8]

        Returns:
            Prediction dictionary
        """
        # Add batch dimension
        batch = {
            "vis_feats": vis_feats.unsqueeze(0),
            "telemetry": telemetry.unsqueeze(0),
        }

        predictions = self.predict(batch)
        return predictions[0]

    def get_label_maps(self) -> Dict[str, Dict[int, str]]:
        """
        Get label ID to name mappings.

        Returns:
            Dictionary with risk, reason, action label maps
        """
        return self.label_maps

    def save_label_maps(self, output_path: str):
        """
        Save label maps to JSON file.

        Args:
            output_path: Output file path
        """
        with open(output_path, "w") as f:
            json.dump(self.label_maps, f, indent=2)
        print(f"Saved label maps to: {output_path}")


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Run SIM v0 inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    args = parser.parse_args()

    # Load model
    sim = SimV0Inference(args.checkpoint, device=args.device)

    # Print label maps
    print("\nLabel Maps:")
    print(json.dumps(sim.get_label_maps(), indent=2))

    # Example inference on dummy data
    print("\nRunning test inference on dummy data...")
    vis_feats = torch.randn(8, 768)
    telemetry = torch.randn(8)

    prediction = sim.predict_single(vis_feats, telemetry)

    print("\nPrediction:")
    print(json.dumps(prediction, indent=2))


if __name__ == "__main__":
    main()
