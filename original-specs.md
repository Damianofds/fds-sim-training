# SIM v0 (Frozen LLM) - Training Code Plan (High Level, No Data Pipeline)

Goal:
- Train a Spatial Intent Model v0 with a **frozen LLM** (and typically frozen vision encoder),
- Produce one or more model checkpoints into an **output folder**,
- Assume training data already exists in a simple, loader-friendly format.

----------------------------------------------------------------------
1) TRAINING INPUT SPEC (what the code expects)
----------------------------------------------------------------------

Your training code will assume a Python `Dataset` yields dicts with:

A) Video representation (choose ONE)
- Option A1: raw frames
  - "frames": float tensor [F, 3, H, W]  (F=8 suggested)
- Option A2: cached vision features (recommended for v0 speed)
  - "vis_feats": float tensor [F, dv]    (dv depends on chosen vision encoder)

B) Minimal telemetry (super-minimal, single token)
- "telemetry": float tensor [8]
  - [ax, ay, az, gx, gy, gz, v, w]
  - normalized (mean/std) before training

C) Labels (classification)
- "risk_id":   int in {0,1,2,3}   (none,low,med,high)
- "reason_id": int in {0,1,2,3}   (human,obstacle,tight_space,unknown)
- "action_id": int in {0..5}      (continue,slow,stop,reverse,turn_left,turn_right)

Optional:
- "weight": float scalar (for class imbalance)
- "sample_id": string (for debugging)

----------------------------------------------------------------------
2) MODEL ARCHITECTURE SPEC (what you implement in code)
----------------------------------------------------------------------

Concept:
- Convert (video + telemetry) into a short prefix of embeddings,
- Feed into a frozen decoder-only LLM via `inputs_embeds`,
- Train small heads for risk/reason/action.

Components:
- Frozen vision encoder (only if using raw frames)
- Trainable vision projector: dv -> d
- Trainable telemetry MLP: 8 -> d
- Trainable task token: [d]
- Frozen LLM: embedding dim d
- Trainable heads:
  - risk head:   d -> 4
  - reason head: d -> 4
  - action head: d -> 6
  - optional confidence head: d -> 1

Fusion:
- Temporal reduce F frame features into K visual tokens (K=4 recommended)
  - simplest: segment-average pooling (no params)
- Create prefix embeddings:
  X = [V_tokens (K), T_token (1), Q_token (1)]  => length L = K + 2
- Run frozen LLM on X
- Readout: last token hidden state (Q position)
- Heads produce logits

Trainable params only:
- vision projector
- telemetry MLP
- task token
- heads

----------------------------------------------------------------------
3) TRAINING OBJECTIVE (losses + metrics)
----------------------------------------------------------------------

Loss:
- L = CE(risk) + 0.5 * CE(reason) + CE(action)
- Optional class weights supported if provided by dataset

Key metrics to print each epoch:
- risk_high_recall (recall for class "high")
- risk_high_fn_rate (false negative rate on "high")
- action_accuracy
- macro_f1 (risk + action) (optional)

----------------------------------------------------------------------
4) TRAINING SCRIPT CONTRACT (CLI + outputs)
----------------------------------------------------------------------

Implement a single entrypoint, e.g.:
- `python train_sim_v0.py --config configs/sim_v0.yaml`

Inputs via config:
- data:
  - dataset_class_path (import path string)
  - train_split_path / val_split_path (user-defined)
  - batch_size
- model:
  - llm_name_or_path
  - vision_encoder_name (optional if raw frames)
  - use_cached_vis_feats (bool)
  - dv, d
  - F (frames/window), K (visual tokens)
- training:
  - epochs
  - lr
  - weight_decay
  - grad_clip
  - amp (true/false)
  - seed
- output:
  - output_dir

Output directory layout (created by code):
output_dir/
  config_resolved.yaml
  checkpoints/
    best.pt
    last.pt
    epoch_001.pt (optional)
  metrics/
    train_metrics.jsonl
    val_metrics.jsonl
  logs/
    stdout.log
  export/
    sim_v0_infer.py (optional minimal inference wrapper)
    labels.json (id->name mappings)

Definition of "model artifact" to save:
- state_dict for trainable modules
- plus a small metadata dict:
  - dv, d, F, K
  - label maps
  - normalization stats keys expected (telemetry mean/std if you want)

----------------------------------------------------------------------
5) IMPLEMENTATION STEPS (what Claude should write)
----------------------------------------------------------------------

A) `SimV0Model` (PyTorch module)
- Accept batch dict:
  - either "vis_feats" OR "frames"
  - plus "telemetry"
- Produce logits dict:
  - risk_logits, reason_logits, action_logits (+ conf)
- Provide helper:
  - `predict_json(batch)` that returns readable JSON objects per sample

B) `train_one_epoch(model, loader, optimizer, scaler, device)`
- forward
- compute loss
- backward (AMP optional)
- step
- compute running metrics

C) `evaluate(model, loader, device)`
- no_grad
- metrics only

D) `checkpointing`
- Save:
  - best on val risk_high_recall (or composite score)
  - last every epoch

E) `reproducibility`
- seed everything
- log config and git hash (if available)

----------------------------------------------------------------------
6) WHAT TO FREEZE (explicitly in code)
----------------------------------------------------------------------

- Freeze LLM parameters:
  for p in llm.parameters(): p.requires_grad = False
- If vision encoder used:
  freeze it too
- Assert trainable parameter count > 0 and matches expected modules

----------------------------------------------------------------------
7) V0 DEFAULTS (good starting knobs)
----------------------------------------------------------------------

- F = 8 frames per sample
- K = 4 visual tokens after temporal pooling
- telemetry = 8 floats -> 1 token
- batch_size = 64 (if cached features) / 8-16 (if raw frames)
- lr = 1e-3 for projector/heads, or single lr=1e-3 for all trainables
- epochs = 10
- optimizer = AdamW
- grad_clip = 1.0

----------------------------------------------------------------------
8) DELIVERABLE (what the training code produces)
----------------------------------------------------------------------

After running training:
- You have at least:
  - output_dir/checkpoints/best.pt
  - output_dir/checkpoints/last.pt
- Plus logs and metrics history for quick iteration.

This is sufficient to:
- load the checkpoint
- run inference on rover streams
- drive a live "risk/action overlay" demo.

----------------------------------------------------------------------
9) NON-GOALS (explicitly excluded for v0 training code plan)
----------------------------------------------------------------------

- No logging/data collection code
- No labeling UI
- No window slicing / synchronization pipeline
- No ROS integration
- No deployment packaging beyond a simple inference wrapper
