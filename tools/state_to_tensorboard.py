"""Convert a Hugging Face trainer_state.json (log_history) into TensorBoard events.

Usage:
    python tools/state_to_tensorboard.py \
        --state checkpoints/qwen35_9b_stemo_ambig_lora_v5/checkpoint-400/trainer_state.json \
        --logdir checkpoints/qwen35_9b_stemo_ambig_lora_v5/runs

Then:
    tensorboard --logdir checkpoints/qwen35_9b_stemo_ambig_lora_v5/runs --port 6006
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--logdir", type=Path, required=True)
    args = ap.parse_args()

    from torch.utils.tensorboard import SummaryWriter
    state = json.loads(args.state.read_text())
    log = state.get("log_history", [])
    print(f"loaded {len(log)} log entries from {args.state}")

    args.logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.logdir))
    n_logged = 0
    for entry in log:
        step = entry.get("step")
        if step is None:
            continue
        for k, v in entry.items():
            if k == "step" or v is None:
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                writer.add_scalar(k, float(v), int(step))
                n_logged += 1
    writer.close()
    print(f"wrote {n_logged} scalar values to {args.logdir}")


if __name__ == "__main__":
    main()
