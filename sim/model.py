"""
SimV0Model: Spatial Intent Model with frozen LLM.

Architecture:
- Frozen vision encoder (optional, if using raw frames)
- Trainable vision projector: dv -> d
- Trainable telemetry MLP: 8 -> d
- Trainable task token
- Frozen LLM (decoder-only)
- Trainable classification heads
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from transformers import AutoModel, AutoConfig


class VisionProjector(nn.Module):
    """Projects vision features to LLM embedding dimension."""

    def __init__(self, dv: int, d: int):
        """
        Args:
            dv: Vision feature dimension
            d: LLM embedding dimension
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dv, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Vision features [B, F, dv]

        Returns:
            Projected features [B, F, d]
        """
        return self.proj(x)


class TelemetryMLP(nn.Module):
    """Projects telemetry to LLM embedding dimension."""

    def __init__(self, input_dim: int = 8, d: int = 768):
        """
        Args:
            input_dim: Telemetry dimension (default 8)
            d: LLM embedding dimension
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Telemetry [B, 8]

        Returns:
            Projected telemetry [B, d]
        """
        return self.mlp(x)


class TemporalPooling(nn.Module):
    """Segment-average pooling to reduce F frames to K tokens."""

    def __init__(self, K: int = 4):
        """
        Args:
            K: Number of output tokens
        """
        super().__init__()
        self.K = K

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features [B, F, d]

        Returns:
            Pooled features [B, K, d]
        """
        B, F, d = x.shape

        # Segment the F frames into K groups and average each
        segment_size = F // self.K
        remainder = F % self.K

        pooled = []
        start = 0
        for i in range(self.K):
            # Distribute remainder across first segments
            end = start + segment_size + (1 if i < remainder else 0)
            segment = x[:, start:end, :]  # [B, segment_len, d]
            pooled.append(segment.mean(dim=1))  # [B, d]
            start = end

        return torch.stack(pooled, dim=1)  # [B, K, d]


class SimV0Model(nn.Module):
    """
    Spatial Intent Model v0 with frozen LLM.

    Combines video and telemetry into a prefix for a frozen LLM,
    then predicts risk, reason, and action from the last token.
    """

    def __init__(
        self,
        llm_name_or_path: str,
        dv: int,
        d: int,
        F: int = 8,
        K: int = 4,
        use_cached_vis_feats: bool = True,
        vision_encoder_name: Optional[str] = None,
    ):
        """
        Args:
            llm_name_or_path: HuggingFace model name or path (e.g., "gpt2")
            dv: Vision feature dimension
            d: LLM embedding dimension
            F: Number of frames per sample
            K: Number of visual tokens after temporal pooling
            use_cached_vis_feats: Whether to use cached features or raw frames
            vision_encoder_name: Vision encoder name (if using raw frames)
        """
        super().__init__()

        self.dv = dv
        self.d = d
        self.F = F
        self.K = K
        self.use_cached_vis_feats = use_cached_vis_feats

        # Load frozen LLM
        print(f"Loading LLM: {llm_name_or_path}")
        try:
            self.llm = AutoModel.from_pretrained(
                llm_name_or_path,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"Warning: Could not load model with AutoModel, trying GPT2Model: {e}")
            from transformers import GPT2Model
            self.llm = GPT2Model.from_pretrained(llm_name_or_path)

        # Freeze LLM
        for param in self.llm.parameters():
            param.requires_grad = False
        print("Frozen LLM parameters")

        # Vision encoder (if using raw frames)
        if not use_cached_vis_feats and vision_encoder_name:
            print(f"Loading vision encoder: {vision_encoder_name}")
            self.vision_encoder = AutoModel.from_pretrained(vision_encoder_name)
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            print("Frozen vision encoder parameters")
        else:
            self.vision_encoder = None

        # Trainable components
        self.vision_projector = VisionProjector(dv, d)
        self.telemetry_mlp = TelemetryMLP(input_dim=8, d=d)
        self.temporal_pooling = TemporalPooling(K=K)

        # Trainable task query token
        self.task_token = nn.Parameter(torch.randn(1, 1, d))

        # Classification heads
        self.risk_head = nn.Linear(d, 4)  # none, low, med, high
        self.reason_head = nn.Linear(d, 4)  # human, obstacle, tight_space, unknown
        self.action_head = nn.Linear(d, 6)  # continue, slow, stop, reverse, turn_left, turn_right

        # Optional confidence head
        self.confidence_head = nn.Linear(d, 1)

        print(f"Initialized SimV0Model (F={F}, K={K}, dv={dv}, d={d})")
        self._print_parameter_count()

    def _print_parameter_count(self):
        """Print trainable and total parameter counts."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {trainable:,}")
        print(f"Total parameters: {total:,}")
        print(f"Frozen parameters: {total - trainable:,}")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            batch: Dictionary containing:
                - "vis_feats" [B, F, dv] OR "frames" [B, F, 3, H, W]
                - "telemetry" [B, 8]

        Returns:
            Dictionary with:
                - "risk_logits" [B, 4]
                - "reason_logits" [B, 4]
                - "action_logits" [B, 6]
                - "confidence" [B, 1]
        """
        device = next(self.parameters()).device

        # Get vision features
        if "vis_feats" in batch:
            vis_feats = batch["vis_feats"].to(device)  # [B, F, dv]
        elif "frames" in batch:
            if self.vision_encoder is None:
                raise ValueError("Vision encoder not initialized but raw frames provided")
            frames = batch["frames"].to(device)  # [B, F, 3, H, W]
            # Process frames through vision encoder
            B, F, C, H, W = frames.shape
            frames_flat = frames.view(B * F, C, H, W)
            with torch.no_grad():
                vis_feats_flat = self.vision_encoder(frames_flat).last_hidden_state.mean(dim=1)  # [B*F, dv]
            vis_feats = vis_feats_flat.view(B, F, -1)  # [B, F, dv]
        else:
            raise ValueError("Batch must contain either 'vis_feats' or 'frames'")

        telemetry = batch["telemetry"].to(device)  # [B, 8]

        # Project vision features
        vis_projected = self.vision_projector(vis_feats)  # [B, F, d]

        # Temporal pooling: F -> K tokens
        vis_tokens = self.temporal_pooling(vis_projected)  # [B, K, d]

        # Project telemetry to token
        telem_token = self.telemetry_mlp(telemetry).unsqueeze(1)  # [B, 1, d]

        # Task query token
        B = vis_tokens.shape[0]
        query_token = self.task_token.expand(B, -1, -1)  # [B, 1, d]

        # Concatenate prefix: [V_tokens (K), T_token (1), Q_token (1)]
        prefix_embeds = torch.cat([vis_tokens, telem_token, query_token], dim=1)  # [B, K+2, d]

        # Run through frozen LLM
        with torch.no_grad():
            llm_output = self.llm(inputs_embeds=prefix_embeds)
            hidden_states = llm_output.last_hidden_state  # [B, K+2, d]

        # Readout from last token (query position)
        readout = hidden_states[:, -1, :]  # [B, d]

        # Classification heads
        risk_logits = self.risk_head(readout)  # [B, 4]
        reason_logits = self.reason_head(readout)  # [B, 4]
        action_logits = self.action_head(readout)  # [B, 6]
        confidence = self.confidence_head(readout)  # [B, 1]

        return {
            "risk_logits": risk_logits,
            "reason_logits": reason_logits,
            "action_logits": action_logits,
            "confidence": confidence,
        }

    @torch.no_grad()
    def predict_json(self, batch: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
        """
        Generate human-readable predictions.

        Args:
            batch: Input batch

        Returns:
            List of prediction dictionaries, one per sample
        """
        from sim.dataset import SimDataset

        self.eval()
        outputs = self.forward(batch)

        risk_preds = outputs["risk_logits"].argmax(dim=1).cpu().numpy()
        reason_preds = outputs["reason_logits"].argmax(dim=1).cpu().numpy()
        action_preds = outputs["action_logits"].argmax(dim=1).cpu().numpy()
        confidence = torch.sigmoid(outputs["confidence"]).squeeze(-1).cpu().numpy()

        results = []
        for i in range(len(risk_preds)):
            result = {
                "risk": SimDataset.RISK_LABELS[int(risk_preds[i])],
                "reason": SimDataset.REASON_LABELS[int(reason_preds[i])],
                "action": SimDataset.ACTION_LABELS[int(action_preds[i])],
                "confidence": float(confidence[i]),
            }
            if "sample_id" in batch:
                result["sample_id"] = batch["sample_id"][i]
            results.append(result)

        return results

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata for checkpoint saving.

        Returns:
            Dictionary with architecture parameters and label maps
        """
        from sim.dataset import SimDataset

        return {
            "dv": self.dv,
            "d": self.d,
            "F": self.F,
            "K": self.K,
            "use_cached_vis_feats": self.use_cached_vis_feats,
            "label_maps": {
                "risk": SimDataset.RISK_LABELS,
                "reason": SimDataset.REASON_LABELS,
                "action": SimDataset.ACTION_LABELS,
            },
        }
