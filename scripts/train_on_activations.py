"""
Train per-experiment linear probes on saved activations and CSV labels.

Inputs per experiment directory under `results/<exp>`:
- `<model>_results_<Mode>.csv` (from processBatchResults.py) with a boolean `correct` column
- Activations saved in RDS6 checkpoints: `/rds-d6/.../checkpoints/<exp>/<image>_activations.pt`

This script trains an L1-regularized logistic probe per experiment to predict `correct`.
It standardizes features, evaluates a holdout split, and writes metrics and top-weight features.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from rds6_config import get_experiment_path, get_model_path


DEFAULT_CATEGORY_KEYS = ["residual_stream"]  # add: "attention_heads", "mlp", "layer_norms"


class ActivationDataset(Dataset):
    def __init__(self, examples: List[Dict[str, Any]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.examples[idx]


def find_results_csv(results_dir: Path, model: str):
    """Return the first existing results CSV among known modes."""
    candidates = [
        results_dir / f"{model}_results_Cells.csv",
        results_dir / f"{model}_results_Quadrant.csv",
        results_dir / f"{model}_results_Presence.csv",
        results_dir / f"{model}_results_Coords.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def resolve_activation_path(checkpoints_dir: Path, filename: str) -> Optional[Path]:
    """Try several patterns to locate the saved activation file.

    Supports both patterns produced by different runs:
    - f"{stem}_activations.pt"
    - f"{name}_activations.pt" (keeps extension like .png in the base name)
    """
    name = Path(filename).name           # e.g., image_0.png
    stem = Path(filename).stem           # e.g., image_0
    candidates = [
        checkpoints_dir / f"{name}_activations.pt",
        checkpoints_dir / f"{stem}_activations.pt",
        checkpoints_dir / f"{stem}.png_activations.pt",
        checkpoints_dir / f"{stem}.jpg_activations.pt",
        checkpoints_dir / f"{stem}.jpeg_activations.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def build_feature_vector(
    activations: Dict[str, Any],
    categories: List[str],
    allowed_layers: Optional[Set[str]] = None,
) -> Tuple[torch.Tensor, List[Tuple[str, str, int]]]:
    """Concatenate hidden vectors from each selected category/layer.

    Returns tensor [feature_dim] and an index map to (category, layer_key, neuron_idx).
    """
    feature_chunks: List[torch.Tensor] = []
    index_map: List[Tuple[str, str, int]] = []

    for category in categories:
        cat_dict: Dict[str, Any] = activations.get(category, {})
        for layer_key in sorted(cat_dict.keys()):
            if allowed_layers is not None and layer_key not in allowed_layers:
                continue
            arr = cat_dict.get(layer_key)
            if arr is None:
                continue
            t = torch.from_numpy(arr)
            # Expect [batch(=1), seq(=1), hidden] or [seq, hidden]
            if t.ndim == 3:
                vec = t[0, 0]  # [hidden]
            elif t.ndim == 2:
                vec = t[0]
            else:
                vec = t.flatten()
            feature_chunks.append(vec)
            for i in range(vec.shape[0]):
                index_map.append((category, layer_key, i))

    if not feature_chunks:
        return torch.empty(0), []
    return torch.cat(feature_chunks, dim=0).float(), index_map


def load_examples_for_experiment(
    exp_name: str,
    model: str,
    categories: List[str],
    allowed_layers: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str, int]]]:
    results_dir = Path("results") / exp_name
    results_csv = find_results_csv(results_dir, model)
    if results_csv is None:
        print(f"Skipping {exp_name}: results CSV not found for model {model}.")
        return [], []

    df = pd.read_csv(results_csv)
    checkpoints_dir = Path(get_experiment_path(exp_name, "checkpoints"))

    examples: List[Dict[str, Any]] = []
    shared_index_map: List[Tuple[str, str, int]] | None = None

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading examples for {exp_name}"):
        filename = str(row.get("filename", ""))
        if not filename:
            continue
        act_path = resolve_activation_path(checkpoints_dir, filename)
        if act_path is None:
            continue

        try:
            activations = torch.load(act_path)
        except Exception as e:
            print(f"Failed to load {act_path}: {e}")
            continue

        x, index_map = build_feature_vector(activations, categories, allowed_layers)
        if x.numel() == 0:
            continue

        # Ensure consistent index_map across all examples
        if shared_index_map is None:
            shared_index_map = index_map

        y = 1.0 if bool(row.get("correct", False)) else 0.0
        examples.append({"x": x, "y": torch.tensor([y], dtype=torch.float32)})

    return examples, (shared_index_map or [])


def get_layer_keys(exp_name: str, model: str, categories: List[str]) -> List[str]:
    """Inspect activations of the first available example to list layer keys."""
    results_dir = Path("results") / exp_name
    results_csv = find_results_csv(results_dir, model)
    if results_csv is None:
        return []
    df = pd.read_csv(results_csv)
    checkpoints_dir = Path(get_experiment_path(exp_name, "checkpoints"))
    for _, row in df.iterrows():
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
        keys: Set[str] = set()
        for cat in categories:
            keys.update(list(activations.get(cat, {}).keys()))
        return sorted(keys)
    return []


class L1LogisticProbe(nn.Module):
    def __init__(self, in_dim: int, l1_lambda: float):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)
        self.l1_lambda = l1_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)

    def l1_penalty(self) -> torch.Tensor:
        return self.l1_lambda * self.linear.weight.abs().sum()


def standardize_split(train_x: torch.Tensor, val_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train_x - mean) / std, (val_x - mean) / std, mean.squeeze(0), std.squeeze(0)


def train_probe(examples: List[Dict[str, Any]], epochs: int, lr: float, l1_lambda: float, batch_size: int, device: torch.device) -> Tuple[L1LogisticProbe, Dict[str, float]]:
    xs = torch.stack([e["x"] for e in examples])
    ys = torch.stack([e["y"] for e in examples]).squeeze(1)

    # Shuffle and split
    idx = np.random.permutation(xs.size(0))
    split = max(1, int(0.8 * xs.size(0)))
    train_idx, val_idx = idx[:split], idx[split:]
    train_x, val_x = xs[train_idx], xs[val_idx]
    train_y, val_y = ys[train_idx], ys[val_idx]

    # Standardize
    train_x, val_x, mean, std = standardize_split(train_x, val_x)

    model = L1LogisticProbe(in_dim=train_x.size(1), l1_lambda=l1_lambda).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    def iterate(x: torch.Tensor, y: torch.Tensor, train: bool) -> float:
        total = 0.0
        n = 0
        for start in range(0, x.size(0), batch_size):
            xb = x[start:start + batch_size].to(device)
            yb = y[start:start + batch_size].to(device)
            logits = model(xb)
            loss = bce(logits, yb) + model.l1_penalty()
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item()); n += 1
        return total / max(n, 1)

    for epoch in tqdm(range(epochs), desc="Training epochs"):
        model.train(); _ = iterate(train_x, train_y, True)
        model.eval(); _ = iterate(val_x, val_y, False)

    # Final metrics
    with torch.no_grad():
        logits = model(val_x.to(device))
        preds = (torch.sigmoid(logits) > 0.5).float().cpu()
        acc = (preds == val_y).float().mean().item()

    metrics = {
        "val_accuracy": round(acc, 4),
        "num_train": int(train_x.size(0)),
        "num_val": int(val_x.size(0)),
        "feature_dim": int(train_x.size(1)),
        "mean": mean.cpu().numpy().tolist(),
        "std": std.cpu().numpy().tolist(),
    }
    return model, metrics


def top_features(model: L1LogisticProbe, index_map: List[Tuple[str, str, int]], k: int = 20) -> List[Dict[str, Any]]:
    w = model.linear.weight.detach().cpu().squeeze(0)
    order = torch.argsort(w.abs(), descending=True)[: min(k, w.numel())]
    out: List[Dict[str, Any]] = []
    for idx in order.tolist():
        cat, layer_key, neuron = index_map[idx]
        out.append({
            "category": cat,
            "layer": layer_key,
            "neuron": int(neuron),
            "weight": float(w[idx].item()),
        })
    return out


def group_importance(model: L1LogisticProbe, index_map: List[Tuple[str, str, int]]) -> List[Dict[str, Any]]:
    """Aggregate |weights| per (category, layer)."""
    w = model.linear.weight.detach().cpu().squeeze(0)
    agg: Dict[Tuple[str, str], float] = {}
    for idx, (cat, layer_key, _) in enumerate(index_map):
        agg[(cat, layer_key)] = agg.get((cat, layer_key), 0.0) + float(w[idx].abs().item())
    items = [
        {"category": cat, "layer": layer, "importance": val}
        for (cat, layer), val in agg.items()
    ]
    items.sort(key=lambda d: d["importance"], reverse=True)
    return items


def stability_selection(
    examples: List[Dict[str, Any]],
    index_map: List[Tuple[str, str, int]],
    epochs: int,
    lr: float,
    l1_lambda: float,
    batch_size: int,
    device: torch.device,
    bootstraps: int = 30,
) -> List[Dict[str, Any]]:
    """Frequency with which each feature is selected (|w|>0) across bootstrap fits."""
    xs = torch.stack([e["x"] for e in examples])
    ys = torch.stack([e["y"] for e in examples]).squeeze(1)
    n = xs.size(0)
    counts = torch.zeros(xs.size(1), dtype=torch.long)

    for _ in tqdm(range(max(1, bootstraps)), desc="Stability selection bootstraps"):
        # sample ~80% with replacement
        idx = np.random.choice(n, size=max(1, int(0.8 * n)), replace=True)
        sub_examples = [{"x": xs[i], "y": ys[i]} for i in idx]
        model, _ = train_probe(sub_examples, epochs, lr, l1_lambda, batch_size, device)
        w = model.linear.weight.detach().cpu().squeeze(0)
        counts += (w.abs() > 1e-8).long()

    freq = counts.float() / float(max(1, bootstraps))
    order = torch.argsort(freq, descending=True)
    out: List[Dict[str, Any]] = []
    for idx in order.tolist():
        cat, layer_key, neuron = index_map[idx]
        out.append({
            "category": cat,
            "layer": layer_key,
            "neuron": int(neuron),
            "selection_freq": float(freq[idx].item()),
        })
    return out


def save_report(exp_name: str, model_name: str, metrics: Dict[str, Any], features: List[Dict[str, Any]]) -> None:
    report = {"model": model_name, "metrics": metrics, "top_features": features}
    out_path = Path(get_model_path(exp_name, f"{model_name}_probe_report.json"))
    out_path.write_text(json.dumps(report, indent=2))
    print(f"✓ Wrote report: {out_path}")


# ---- Non-linear probe variants ----
class MLPProbeSmall(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class MLPProbeTiny(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def train_probe_mlp(
    examples: List[Dict[str, Any]],
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    variant: str = "small",
    weight_decay: float = 0.0,
) -> Tuple[nn.Module, Dict[str, float]]:
    xs = torch.stack([e["x"] for e in examples])
    ys = torch.stack([e["y"] for e in examples]).squeeze(1)

    idx = np.random.permutation(xs.size(0))
    split = max(1, int(0.8 * xs.size(0)))
    train_idx, val_idx = idx[:split], idx[split:]
    train_x, val_x = xs[train_idx], xs[val_idx]
    train_y, val_y = ys[train_idx], ys[val_idx]

    # Standardize
    train_x, val_x, mean, std = standardize_split(train_x, val_x)

    in_dim = train_x.size(1)
    model: nn.Module = MLPProbeSmall(in_dim) if variant == "small" else MLPProbeTiny(in_dim)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()

    def iterate(x: torch.Tensor, y: torch.Tensor, train: bool) -> float:
        total = 0.0; n = 0
        for start in range(0, x.size(0), batch_size):
            xb = x[start:start + batch_size].to(device)
            yb = y[start:start + batch_size].to(device)
            logits = model(xb)
            loss = bce(logits, yb)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item()); n += 1
        return total / max(n, 1)

    for _ in tqdm(range(epochs), desc=f"Training MLP-{variant}"):
        model.train(); _ = iterate(train_x, train_y, True)
        model.eval(); _ = iterate(val_x, val_y, False)

    with torch.no_grad():
        logits = model(val_x.to(device))
        preds = (torch.sigmoid(logits) > 0.5).float().cpu()
        acc = (preds == val_y).float().mean().item()

    metrics = {
        "val_accuracy": round(acc, 4),
        "num_train": int(train_x.size(0)),
        "num_val": int(val_x.size(0)),
        "feature_dim": int(train_x.size(1)),
        "mean": mean.cpu().numpy().tolist(),
        "std": std.cpu().numpy().tolist(),
        "probe_type": f"mlp_{variant}",
    }
    return model, metrics


def grid_search_probe(
    examples: List[Dict[str, Any]],
    l1_grid: List[float],
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
) -> Tuple[L1LogisticProbe, Dict[str, Any], float]:
    """Train multiple probes across an L1 grid and return the best by val_accuracy."""
    best_model: Optional[L1LogisticProbe] = None
    best_metrics: Optional[Dict[str, Any]] = None
    best_l1: float = l1_grid[0] if l1_grid else 1e-4
    best_acc: float = -1.0
    for l1 in l1_grid:
        model, metrics = train_probe(examples, epochs, lr, l1, batch_size, device)
        acc = float(metrics.get("val_accuracy", 0.0))
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_metrics = metrics
            best_l1 = l1
    assert best_model is not None and best_metrics is not None
    best_metrics = dict(best_metrics)
    best_metrics["best_l1"] = best_l1
    return best_model, best_metrics, best_l1


def load_examples_for_experiments(
    experiments: List[str],
    model: str,
    categories: List[str],
    allowed_layers: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str, int]]]:
    """Pool examples from multiple experiments."""
    pooled: List[Dict[str, Any]] = []
    shared_index_map: List[Tuple[str, str, int]] | None = None
    for exp in experiments:
        ex, idx = load_examples_for_experiment(exp, model, categories, allowed_layers)
        if ex:
            pooled.extend(ex)
            if shared_index_map is None:
                shared_index_map = idx
    return pooled, (shared_index_map or [])


def write_summary_csv(rows: List[Dict[str, Any]], model_name: str, filename: str = None) -> Path:
    df = pd.DataFrame(rows)
    if filename is None:
        filename = f"{model_name}_battery_summary.csv"
    out_path = Path(get_model_path("JOINT", filename))
    df.to_csv(out_path, index=False)
    print(f"✓ Wrote summary CSV: {out_path}")
    return out_path


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
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l1", type=float, default=1e-4, help="L1 regularization strength")
    parser.add_argument("--per_layer", action="store_true", help="Train one probe per layer")
    parser.add_argument("--per_category", action="store_true", help="Train one probe per category")
    parser.add_argument("--stability", type=int, default=0, help="If >0, run stability selection with this many bootstraps")
    parser.add_argument("--battery", action="store_true", help="Run a full battery: combined, per-layer, per-category, joint")
    parser.add_argument("--l1_grid", nargs="*", type=float, default=[1e-3, 5e-4, 1e-4, 5e-5, 1e-5], help="Grid of L1 strengths")
    parser.add_argument("--nonlinear", action="store_true", help="Also evaluate two small MLP probes (small/tiny)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.battery:
        # Original single-run behavior (with optional per_* and stability), but with l1 from --l1
        for exp in tqdm(args.experiments, desc="Processing experiments"):
            print(f"\n=== Experiment: {exp} ===")
            examples, index_map = load_examples_for_experiment(exp, args.model, args.categories)
            if not examples:
                print("No examples found; skipping.")
                continue
            model, metrics, best_l1 = grid_search_probe(examples, args.l1_grid or [args.l1], args.epochs, args.lr, args.batch_size, device)
            feats = top_features(model, index_map, k=30)
            groups = group_importance(model, index_map)
            stab = stability_selection(examples, index_map, args.epochs, args.lr, best_l1, args.batch_size, device, bootstraps=args.stability) if args.stability > 0 else []
            metrics["groups_top10"] = groups[:10]
            print(f"val_accuracy={metrics['val_accuracy']:.3f}  best_l1={metrics.get('best_l1')}  features={len(index_map)}")
            report = {"model": args.model, "metrics": metrics, "top_features": feats, "group_importance": groups, "stability": stab}
            out_path = Path(get_model_path(exp, f"{args.model}_probe_report.json"))
            out_path.write_text(json.dumps(report, indent=2))
            print(f"✓ Wrote report: {out_path}")

            if args.nonlinear:
                for variant in ("small", "tiny"):
                    m_mlp, met_mlp = train_probe_mlp(examples, args.epochs, args.lr, args.batch_size, device, variant=variant)
                    rep = {"model": args.model, "metrics": met_mlp, "probe_type": met_mlp.get("probe_type")}
                    out_mlp = Path(get_model_path(exp, f"{args.model}_probe_{met_mlp['probe_type']}.json"))
                    out_mlp.write_text(json.dumps(rep, indent=2))

            if args.per_layer:
                layer_keys = get_layer_keys(exp, args.model, args.categories)
                for layer_key in tqdm(layer_keys, desc="Training per-layer probes"):
                    allowed = {layer_key}
                    ex_layer, idx_map_layer = load_examples_for_experiment(exp, args.model, args.categories, allowed_layers=allowed)
                    if not ex_layer:
                        continue
                    m_layer, met_layer, best_l1_layer = grid_search_probe(ex_layer, args.l1_grid or [args.l1], args.epochs, args.lr, args.batch_size, device)
                    feats_layer = top_features(m_layer, idx_map_layer, k=20)
                    groups_layer = group_importance(m_layer, idx_map_layer)
                    met_layer["best_l1"] = best_l1_layer
                    rep_layer = {"model": args.model, "layer": layer_key, "metrics": met_layer, "top_features": feats_layer, "group_importance": groups_layer}
                    out_layer = Path(get_model_path(exp, f"{args.model}_probe_{layer_key}.json"))
                    out_layer.write_text(json.dumps(rep_layer, indent=2))
                    if args.nonlinear:
                        for variant in ("small", "tiny"):
                            m_mlp_l, met_mlp_l = train_probe_mlp(ex_layer, args.epochs, args.lr, args.batch_size, device, variant=variant)
                            rep_mlp_l = {"model": args.model, "layer": layer_key, "metrics": met_mlp_l, "probe_type": met_mlp_l.get("probe_type")}
                            out_mlp_l = Path(get_model_path(exp, f"{args.model}_probe_{layer_key}_{met_mlp_l['probe_type']}.json"))
                            out_mlp_l.write_text(json.dumps(rep_mlp_l, indent=2))

            if args.per_category:
                for cat in tqdm(args.categories, desc="Training per-category probes"):
                    ex_cat, idx_map_cat = load_examples_for_experiment(exp, args.model, [cat])
                    if not ex_cat:
                        continue
                    m_cat, met_cat, best_l1_cat = grid_search_probe(ex_cat, args.l1_grid or [args.l1], args.epochs, args.lr, args.batch_size, device)
                    feats_cat = top_features(m_cat, idx_map_cat, k=20)
                    groups_cat = group_importance(m_cat, idx_map_cat)
                    met_cat["best_l1"] = best_l1_cat
                    rep_cat = {"model": args.model, "category": cat, "metrics": met_cat, "top_features": feats_cat, "group_importance": groups_cat}
                    out_cat = Path(get_model_path(exp, f"{args.model}_probe_{cat}.json"))
                    out_cat.write_text(json.dumps(rep_cat, indent=2))
                    if args.nonlinear:
                        for variant in ("small", "tiny"):
                            m_mlp_c, met_mlp_c = train_probe_mlp(ex_cat, args.epochs, args.lr, args.batch_size, device, variant=variant)
                            rep_mlp_c = {"model": args.model, "category": cat, "metrics": met_mlp_c, "probe_type": met_mlp_c.get("probe_type")}
                            out_mlp_c = Path(get_model_path(exp, f"{args.model}_probe_{cat}_{met_mlp_c['probe_type']}.json"))
                            out_mlp_c.write_text(json.dumps(rep_mlp_c, indent=2))

        print("\n✅ Completed training probes for selected experiments.")
        return

    # Battery mode
    summary_rows: List[Dict[str, Any]] = []

    for exp in tqdm(args.experiments, desc="Battery: per-experiment"):
        # Combined categories with grid search
        ex_all, idx_all = load_examples_for_experiment(exp, args.model, args.categories)
        if ex_all:
            model_all, met_all, best_l1_all = grid_search_probe(ex_all, args.l1_grid, args.epochs, args.lr, args.batch_size, device)
            feats_all = top_features(model_all, idx_all, k=30)
            groups_all = group_importance(model_all, idx_all)
            report_all = {"model": args.model, "metrics": met_all, "top_features": feats_all, "group_importance": groups_all}
            Path(get_model_path(exp, f"{args.model}_probe_report.json")).write_text(json.dumps(report_all, indent=2))
            summary_rows.append({
                "experiment": exp, "test": "combined", "identifier": "all", "val_accuracy": met_all.get("val_accuracy"),
                "num_train": met_all.get("num_train"), "num_val": met_all.get("num_val"), "feature_dim": met_all.get("feature_dim"), "best_l1": best_l1_all
            })
            if args.nonlinear:
                for variant in ("small", "tiny"):
                    m_mlp_a, met_mlp_a = train_probe_mlp(ex_all, args.epochs, args.lr, args.batch_size, device, variant=variant)
                    rep_mlp_a = {"model": args.model, "metrics": met_mlp_a, "probe_type": met_mlp_a.get("probe_type")}
                    Path(get_model_path(exp, f"{args.model}_probe_{met_mlp_a['probe_type']}.json")).write_text(json.dumps(rep_mlp_a, indent=2))
                    summary_rows.append({
                        "experiment": exp, "test": met_mlp_a["probe_type"], "identifier": "all", "val_accuracy": met_mlp_a.get("val_accuracy"),
                        "num_train": met_mlp_a.get("num_train"), "num_val": met_mlp_a.get("num_val"), "feature_dim": met_mlp_a.get("feature_dim")
                    })

        # Per-layer
        layer_keys = get_layer_keys(exp, args.model, args.categories)
        for layer_key in layer_keys:
            allowed = {layer_key}
            ex_layer, idx_layer = load_examples_for_experiment(exp, args.model, args.categories, allowed_layers=allowed)
            if not ex_layer:
                continue
            m_layer, met_layer, best_l1_layer = grid_search_probe(ex_layer, args.l1_grid, args.epochs, args.lr, args.batch_size, device)
            feats_layer = top_features(m_layer, idx_layer, k=20)
            groups_layer = group_importance(m_layer, idx_layer)
            rep_layer = {"model": args.model, "layer": layer_key, "metrics": met_layer, "top_features": feats_layer, "group_importance": groups_layer}
            Path(get_model_path(exp, f"{args.model}_probe_{layer_key}.json")).write_text(json.dumps(rep_layer, indent=2))
            summary_rows.append({
                "experiment": exp, "test": "per_layer", "identifier": layer_key, "val_accuracy": met_layer.get("val_accuracy"),
                "num_train": met_layer.get("num_train"), "num_val": met_layer.get("num_val"), "feature_dim": met_layer.get("feature_dim"), "best_l1": best_l1_layer
            })

        # Per-category
        for cat in args.categories:
            ex_cat, idx_cat = load_examples_for_experiment(exp, args.model, [cat])
            if not ex_cat:
                continue
            m_cat, met_cat, best_l1_cat = grid_search_probe(ex_cat, args.l1_grid, args.epochs, args.lr, args.batch_size, device)
            feats_cat = top_features(m_cat, idx_cat, k=20)
            groups_cat = group_importance(m_cat, idx_cat)
            rep_cat = {"model": args.model, "category": cat, "metrics": met_cat, "top_features": feats_cat, "group_importance": groups_cat}
            Path(get_model_path(exp, f"{args.model}_probe_{cat}.json")).write_text(json.dumps(rep_cat, indent=2))
            summary_rows.append({
                "experiment": exp, "test": "per_category", "identifier": cat, "val_accuracy": met_cat.get("val_accuracy"),
                "num_train": met_cat.get("num_train"), "num_val": met_cat.get("num_val"), "feature_dim": met_cat.get("feature_dim"), "best_l1": best_l1_cat
            })

    # Joint probe across experiments (combined categories)
    joint_examples, joint_index = load_examples_for_experiments(args.experiments, args.model, args.categories)
    if joint_examples:
        m_joint, met_joint, best_l1_joint = grid_search_probe(joint_examples, args.l1_grid, args.epochs, args.lr, args.batch_size, device)
        feats_joint = top_features(m_joint, joint_index, k=40)
        groups_joint = group_importance(m_joint, joint_index)
        rep_joint = {"model": args.model, "experiments": args.experiments, "metrics": met_joint, "top_features": feats_joint, "group_importance": groups_joint}
        Path(get_model_path("JOINT", f"{args.model}_probe_joint.json")).write_text(json.dumps(rep_joint, indent=2))
        summary_rows.append({
            "experiment": "JOINT", "test": "combined", "identifier": "all", "val_accuracy": met_joint.get("val_accuracy"),
            "num_train": met_joint.get("num_train"), "num_val": met_joint.get("num_val"), "feature_dim": met_joint.get("feature_dim"), "best_l1": best_l1_joint
        })
        if args.nonlinear:
            for variant in ("small", "tiny"):
                m_mlp_j, met_mlp_j = train_probe_mlp(joint_examples, args.epochs, args.lr, args.batch_size, device, variant=variant)
                rep_mlp_j = {"model": args.model, "experiments": args.experiments, "metrics": met_mlp_j, "probe_type": met_mlp_j.get("probe_type")}
                Path(get_model_path("JOINT", f"{args.model}_probe_joint_{met_mlp_j['probe_type']}.json")).write_text(json.dumps(rep_mlp_j, indent=2))
                summary_rows.append({
                    "experiment": "JOINT", "test": met_mlp_j["probe_type"], "identifier": "all", "val_accuracy": met_mlp_j.get("val_accuracy"),
                    "num_train": met_mlp_j.get("num_train"), "num_val": met_mlp_j.get("num_val"), "feature_dim": met_mlp_j.get("feature_dim")
                })

    # Write summary
    write_summary_csv(summary_rows, args.model)
    print("\n✅ Battery complete.")


if __name__ == "__main__":
    main()

