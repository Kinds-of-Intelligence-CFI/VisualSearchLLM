#!/usr/bin/env python3
"""
Compute linear-probe attributions over saved activation files.

Inputs:
- Probe artifact .pt saved by scripts/train_on_activations.py (contains state_dict, mean/std, index_map, categories, allowed_layers)
- A list of experiments and model name to locate activation files and results CSVs

Outputs:
- JSON per experiment with per-(category, layer) contributions aggregated across examples
- Optional per-example outputs (top contributing (category, layer) pairs)
- Optional zero-out analysis of top-k weighted features in probe space
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from rds6_config import get_experiment_path, get_model_path
from scripts.train_on_activations import (
    DEFAULT_CATEGORY_KEYS,
    find_results_csv,
    resolve_activation_path,
    build_feature_vector,
)
import pandas as pd


def load_probe_artifact(path: Path) -> Dict[str, Any]:
    data = torch.load(path, map_location="cpu")
    return data


def index_map_to_groups(index_map: List[Tuple[str, str, int]]) -> Tuple[List[Tuple[str, str]], Dict[Tuple[str, str], List[int]]]:
    groups: Dict[Tuple[str, str], List[int]] = {}
    for idx, (cat, layer_key, _neuron) in enumerate(index_map):
        groups.setdefault((cat, layer_key), []).append(idx)
    keys = list(groups.keys())
    return keys, groups


def standardize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    if mean.numel() == 0 or std.numel() == 0:
        return x
    return (x - mean) / std.clamp_min(1e-6)


def compute_contributions(
    x_std: torch.Tensor,
    weight: torch.Tensor,
    bias: float,
    index_map: List[Tuple[str, str, int]],
) -> Tuple[float, Dict[Tuple[str, str], float]]:
    # x_std, weight: [D]
    z = float(torch.dot(weight, x_std).item() + bias)
    keys, groups = index_map_to_groups(index_map)
    contrib: Dict[Tuple[str, str], float] = {}
    for key in keys:
        idxs = groups[key]
        # Sum contributions in standardized space: w_i * x_i
        val = float((weight[idxs] * x_std[idxs]).sum().item())
        contrib[key] = val
    return z, contrib


def zero_out_topk(
    x_std: torch.Tensor,
    weight: torch.Tensor,
    k: int,
) -> torch.Tensor:
    # Zero out by |w| rank
    D = weight.numel()
    if k <= 0 or k >= D:
        mask = torch.zeros_like(weight, dtype=torch.bool)
    else:
        order = torch.argsort(weight.abs(), descending=True)
        mask = torch.zeros(D, dtype=torch.bool)
        mask[order[:k]] = True
    x_mod = x_std.clone()
    x_mod[mask] = 0.0
    return x_mod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["llama11B", "llama90B"], default="llama11B")
    parser.add_argument("--experiments", nargs="*",
                        default=[
                            "2Among5ColourRand", "2Among5NoColourRand", "2Among5ConjRand",
                            "LitSpheresTop", "LitSpheresBottom", "LitSpheresLeft", "LitSpheresRight",
                            "CircleSizesSmall", "CircleSizesMedium", "CircleSizesLarge",
                        ])
    parser.add_argument("--categories", nargs="*", default=DEFAULT_CATEGORY_KEYS)
    parser.add_argument("--artifact", help="Path to probe artifact .pt (if omitted, will look up per-experiment combined artifact)")
    parser.add_argument("--per_example", action="store_true", help="Emit per-example contributions")
    parser.add_argument("--zero_topk", type=int, default=0, help="If >0, evaluate effect of zeroing top-k |w| features")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for exp in tqdm(args.experiments, desc="Attribution per experiment"):
        # Locate artifact
        if args.artifact:
            artifact_path = Path(args.artifact)
        else:
            # default to combined artifact path saved by training
            artifact_path = Path(get_model_path(exp, f"{args.model}_probe_combined.pt"))
        if not artifact_path.exists():
            print(f"Skipping {exp}: artifact not found at {artifact_path}")
            continue
        art = load_probe_artifact(artifact_path)
        state = art.get("state_dict")
        mean = art.get("mean", torch.tensor([]))
        std = art.get("std", torch.tensor([]))
        index_map: List[Tuple[str, str, int]] = art.get("index_map", [])
        categories: List[str] = art.get("categories", args.categories)
        allowed_layers = set(art.get("allowed_layers") or []) if art.get("allowed_layers") else None

        # Rebuild weight vector
        # state["linear.weight"] shape [1, D]
        w = torch.tensor(state["linear.weight"]).detach().cpu().squeeze(0)
        b = float(torch.tensor(state["linear.bias"]).detach().cpu().squeeze().item())

        # Load dataset examples directly to retain filenames
        results_dir = Path("results") / exp
        results_csv = find_results_csv(results_dir, args.model)
        if results_csv is None:
            print(f"No results CSV for {exp}")
            continue
        df = pd.read_csv(results_csv)
        checkpoints_dir = Path(get_experiment_path(exp, "checkpoints"))

        # Aggregate contributions
        per_group_sum: Dict[Tuple[str, str], float] = {}
        per_group_count: Dict[Tuple[str, str], int] = {}
        logits: List[float] = []
        logits_zero: List[float] = []
        corrects: List[int] = []
        per_example: List[Dict[str, Any]] = []
        processed_examples: int = 0

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Attr {exp}"):
            filename = str(row.get("filename", ""))
            if not filename:
                continue
            act_path = resolve_activation_path(checkpoints_dir, filename)
            if act_path is None:
                continue
            try:
                activations = torch.load(act_path)
            except Exception:
                continue
            x, idx_map_check = build_feature_vector(activations, categories, allowed_layers=allowed_layers)
            if x.numel() == 0:
                continue
            # optional: sanity on index map length
            if index_map and (len(idx_map_check) != len(index_map)):
                # fallback: skip inconsistent shapes
                continue
            y = int(bool(row.get("correct", False)))
            x_std = standardize(x, mean, std)
            z, contrib = compute_contributions(x_std, w, b, index_map)
            logits.append(z)
            corrects.append(y)
            for key, val in contrib.items():
                per_group_sum[key] = per_group_sum.get(key, 0.0) + val
                per_group_count[key] = per_group_count.get(key, 0) + 1

            if args.zero_topk > 0:
                x0 = zero_out_topk(x_std, w, args.zero_topk)
                z0 = float(torch.dot(w, x0).item() + b)
                logits_zero.append(z0)

            if args.per_example:
                # top-10 groups by absolute contribution
                top = sorted(
                    [
                        {"category": k[0], "layer": k[1], "contribution": float(v)}
                        for k, v in contrib.items()
                    ],
                    key=lambda d: abs(d["contribution"]), reverse=True
                )[:10]
                per_example.append({
                    "filename": filename,
                    "label": y,
                    "logit": z,
                    "top_groups": top,
                })
            processed_examples += 1

        # Build report
        contrib_avg = [
            {"category": cat, "layer": layer, "avg_contribution": per_group_sum[(cat, layer)] / max(1, per_group_count[(cat, layer)])}
            for (cat, layer) in per_group_sum.keys()
        ]
        contrib_avg.sort(key=lambda d: abs(d["avg_contribution"]), reverse=True)

        out: Dict[str, Any] = {
            "model": args.model,
            "artifact": str(artifact_path),
            "experiments": [exp],
            "avg_group_contributions": contrib_avg,
            "num_examples": int(processed_examples),
        }

        if args.zero_topk > 0:
            out["zero_topk"] = args.zero_topk
            out["logits_before_mean"] = float(np.mean(logits)) if logits else 0.0
            out["logits_after_mean"] = float(np.mean(logits_zero)) if logits_zero else 0.0

        # Write JSON
        if args.per_example:
            out["examples"] = per_example
        out_path = Path(get_model_path(exp, f"{args.model}_probe_attribution.json"))
        out_path.write_text(json.dumps(out, indent=2))
        print(f"✓ Wrote attribution: {out_path}")


if __name__ == "__main__":
    main()


