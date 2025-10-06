#!/usr/bin/env python3
"""
Summarize per-(category, layer) probe results into a CSV.

Scans an input models root (e.g., /home/mm2833/VisualSearchLLM/local_models) for files like:
- <model>_probe_<category>_<layer>.json                 (linear)
- <model>_probe_<category>_<layer>_mlp_small.json       (non-linear)
- <model>_probe_<category>_<layer>_mlp_tiny.json        (non-linear)

Outputs a CSV with columns:
  experiment, model, category, layer, probe_type, val_accuracy, num_train, num_val, feature_dim, best_l1
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# Allow categories with underscores by anchoring on the `_layer_<N>` boundary
PATTERN = re.compile(r"^(?P<model>[^_]+)_probe_(?P<category>.+)_(?P<layer>layer_\d+)(?:_(?P<probe>mlp_small|mlp_tiny))?\.json$")


def find_rows(models_root: Path) -> List[Dict]:
    rows: List[Dict] = []
    for exp_dir in sorted([p for p in models_root.iterdir() if p.is_dir()]):
        # skip JOINT if present; focus on per-experiment dirs
        if exp_dir.name.upper() == "JOINT":
            continue
        for f in exp_dir.iterdir():
            if not f.name.endswith(".json"):
                continue
            m = PATTERN.match(f.name)
            if not m:
                continue
            gd = m.groupdict()
            # Prefer model name from JSON payload if present
            layer = gd["layer"]
            probe_type = gd.get("probe") or "linear_l1"
            try:
                data = json.loads(f.read_text())
            except Exception:
                continue
            model = data.get("model") or gd["model"]
            category = gd["category"]
            metrics = data.get("metrics", {})
            row = {
                "experiment": exp_dir.name,
                "model": model,
                "category": category,
                "layer": layer,
                "probe_type": data.get("probe_type", probe_type),
                "val_accuracy": metrics.get("val_accuracy"),
                "num_train": metrics.get("num_train"),
                "num_val": metrics.get("num_val"),
                "feature_dim": metrics.get("feature_dim"),
                "best_l1": metrics.get("best_l1"),
                "source_file": str(f),
            }
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_root", default="/home/mm2833/VisualSearchLLM/local_models", help="Root directory containing per-experiment model outputs")
    parser.add_argument("--out", default=None, help="Output CSV path (defaults to <models_root>/cat_layer_summary.csv)")
    args = parser.parse_args()

    models_root = Path(args.models_root)
    out_path = Path(args.out) if args.out else models_root / "cat_layer_summary.csv"
    rows = find_rows(models_root)
    df = pd.DataFrame(rows)
    # Sort for readability
    if not df.empty:
        df = df.sort_values(["experiment", "category", "layer", "probe_type"]).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"✓ Wrote: {out_path}  ({len(df)} rows)")


if __name__ == "__main__":
    main()


