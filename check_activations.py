import argparse
import os
import glob
import json
from typing import Any, Dict

import numpy as np
import torch


EXPECTED_SECTIONS = [
    "residual_stream",
    "attention_heads",
    "mlp",
    "layer_norms",
    "vision_local_residual",
    "vision_global_residual",
    "projector_input",
    "projector_output",
]


def summarize_array(x: Any) -> str:
    try:
        if isinstance(x, np.ndarray):
            arr = x
        elif torch.is_tensor(x):
            arr = x.detach().cpu().float().numpy()
        else:
            return f"type={type(x)}"
        return (
            f"shape={tuple(arr.shape)}, dtype={arr.dtype}, "
            f"mean={float(arr.mean()):.4f}, std={float(arr.std()):.4f}, "
            f"min={float(arr.min()):.4f}, max={float(arr.max()):.4f}"
        )
    except Exception as e:
        return f"unavailable ({e})"


def inspect_activation_file(path: str) -> None:
    print(f"\n=== {os.path.basename(path)} ===")
    try:
        data: Dict[str, Dict[str, Any]] = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    top_keys = list(data.keys())
    print("Sections:", top_keys)

    # Check presence of expected sections
    missing = [k for k in EXPECTED_SECTIONS if k not in data]
    if missing:
        print("Missing sections:", missing)

    # Print brief stats for a few entries per section
    for section in EXPECTED_SECTIONS:
        sub = data.get(section)
        if not isinstance(sub, dict):
            continue
        keys = list(sub.keys())
        print(f"[{section}] count={len(keys)} keys={keys[:3]}{'...' if len(keys) > 3 else ''}")
        for k in keys[:3]:
            v = sub.get(k)
            if v is None:
                print(f"  {k}: None")
                continue
            print(f"  {k}: {summarize_array(v)}")

    # Attempt to load sibling JSON summary
    if path.endswith("_activations.pt"):
        summary_path = path.replace("_activations.pt", "_activation_summary.json")
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r") as f:
                    summary = json.load(f)
                stats = summary.get("activation_stats", {})
                print("Summary found. Stats sections:", list(stats.keys()))
            except Exception as e:
                print(f"Failed to read summary: {e}")
        else:
            print("Summary JSON not found:", summary_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a few activation files for sanity checks.")
    parser.add_argument(
        "--dir",
        default="/rds-d6/user/mm2833/hpc-work/VisualSearchLLM/checkpoints/2Among5ColourRand",
        help="Directory containing *_activations.pt files",
    )
    parser.add_argument("--limit", type=int, default=3, help="Number of files to inspect")
    parser.add_argument(
        "--pattern", default="*_activations.pt", help="Glob pattern to match activation files"
    )
    args = parser.parse_args()

    glob_pattern = os.path.join(args.dir, args.pattern)
    files = sorted(glob.glob(glob_pattern))
    if not files:
        print("No activation files found at:", glob_pattern)
        return

    for path in files[: args.limit]:
        inspect_activation_file(path)


if __name__ == "__main__":
    main()



