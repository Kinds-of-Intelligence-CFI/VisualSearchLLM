import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score
import matplotlib.pyplot as plt

from rds6_config import get_experiment_path
from scripts.train_on_activations import (
    load_examples_for_experiment,
    build_feature_vector,
    resolve_activation_path,
)


def read_annotations(local_results_dir: Path) -> pd.DataFrame:
    csv_path = local_results_dir / "annotations.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"annotations.csv not found at {csv_path}")
    df = pd.read_csv(csv_path)
    # expected columns: filename, target(bool), center_x, center_y, size, quadrant, num_distractors, color_bin_index
    required = ["filename", "target", "center_x", "center_y", "size", "quadrant", "num_distractors"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required annotation columns: {missing}")
    return df


def add_size_class(df: pd.DataFrame, bins: List[float], labels: List[str]) -> pd.DataFrame:
    df = df.copy()
    # Drop duplicate edges if any
    df["size_class"] = pd.cut(df["size"], bins=bins, labels=labels, include_lowest=True, duplicates="drop")
    return df


def compute_global_size_bins(pooled_sizes: List[float]) -> Tuple[List[float], List[str]]:
    sizes = np.array(pooled_sizes, dtype=float)
    labels = ["Small", "Medium", "Large"]
    uniq = np.unique(sizes)
    if uniq.size >= 3:
        if uniq.size == 3:
            # Midpoints between the three distinct sizes
            s1, s2, s3 = np.sort(uniq)
            b1 = (s1 + s2) / 2.0
            b2 = (s2 + s3) / 2.0
            bins = [-np.inf, b1, b2, np.inf]
        else:
            q1, q2 = np.quantile(sizes, [0.33, 0.66])
            # Ensure strictly increasing
            if q1 >= q2:
                eps = max(1e-6, 1e-6 * (np.max(sizes) - np.min(sizes) + 1.0))
                q2 = q1 + eps
            bins = [-np.inf, float(q1), float(q2), np.inf]
    elif uniq.size == 2:
        # Two bins; fall back to Small/Large only
        s1, s2 = np.sort(uniq)
        mid = (s1 + s2) / 2.0
        bins = [-np.inf, mid, np.inf]
        labels = ["Small", "Large"]
    else:
        # Single size; degenerate – treat all as Medium
        bins = [-np.inf, np.inf]
        labels = ["Medium"]
    return bins, labels


def residualize_against_log_n(features: np.ndarray, log_n: np.ndarray) -> np.ndarray:
    # Regress each feature on log_n and return residuals
    X = log_n.reshape(-1, 1)
    X = np.hstack([np.ones_like(X), X])
    # Closed-form OLS per feature: beta = (X^T X)^{-1} X^T y
    XtX_inv = np.linalg.pinv(X.T @ X)
    betas = XtX_inv @ X.T @ features
    fitted = X @ betas
    resid = features - fitted
    return resid


def compute_bin_means(features: np.ndarray, classes: List[str], class_series: pd.Series) -> Dict[str, np.ndarray]:
    means: Dict[str, np.ndarray] = {}
    for cl in classes:
        idx = class_series == cl
        if idx.any():
            means[cl] = features[idx].mean(axis=0)
    return means


def construct_salience_axis(means: Dict[str, np.ndarray]) -> np.ndarray:
    # Ordinal: (L - M) + (M - S) fallback to (L - S)
    if all(k in means for k in ["Small", "Medium", "Large"]):
        w = (means["Large"] - means["Medium"]) + (means["Medium"] - means["Small"])
    elif all(k in means for k in ["Small", "Large"]):
        w = means["Large"] - means["Small"]
    else:
        raise ValueError("Insufficient class means to construct salience axis")
    # Normalize
    norm = np.linalg.norm(w) + 1e-8
    return w / norm


def project_out_axis(features: np.ndarray, w: np.ndarray) -> np.ndarray:
    # Remove component along w
    comp = features @ w
    return features - np.outer(comp, w)


def fit_predict_location(features: np.ndarray, coords: np.ndarray) -> Tuple[LinearRegression, float]:
    reg = LinearRegression()
    reg.fit(features, coords)
    preds = reg.predict(features)
    r2 = r2_score(coords, preds, multioutput="variance_weighted")
    return reg, float(r2)


def fit_predict_quadrant(features: np.ndarray, quads: np.ndarray) -> Tuple[LogisticRegression, float]:
    lr = LogisticRegression(max_iter=2000, multi_class="multinomial", solver="lbfgs")
    lr.fit(features, quads)
    acc = float(accuracy_score(quads, lr.predict(features)))
    return lr, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", required=True, help="Experiment names under results/")
    parser.add_argument("--model", required=True, help="Model name used in results CSVs")
    parser.add_argument("--categories", nargs="+", default=["residual_stream"], help="Activation categories to include")
    parser.add_argument("--layers", nargs="*", default=None, help="Optional allowed layer keys, e.g., layer_5 layer_50")
    parser.add_argument("--output", default="popout_item_probe_report.json", help="Output JSON report path")
    args = parser.parse_args()

    # First pass: gather annotations to compute global size bins
    exp_to_ann_tgt: Dict[str, pd.DataFrame] = {}
    pooled_sizes: List[float] = []
    for exp in args.experiments:
        local_results_dir = Path("results") / exp
        ann = read_annotations(local_results_dir)
        ann_tgt = ann[ann["target"] == True].copy()
        if not ann_tgt.empty:
            exp_to_ann_tgt[exp] = ann_tgt
            pooled_sizes.extend(ann_tgt["size"].astype(float).tolist())

    if not pooled_sizes:
        raise RuntimeError("No target annotations found across experiments; cannot compute size bins.")

    bins, size_labels = compute_global_size_bins(pooled_sizes)

    # Second pass: load features and merge with size classes using global bins
    all_rows: List[Dict[str, Any]] = []
    index_map: Optional[List[Tuple[str, str, int]]] = None

    for exp in args.experiments:
        ann_tgt = exp_to_ann_tgt.get(exp)
        if ann_tgt is None:
            print(f"No target annotations for {exp}; skipping")
            continue
        ann_tgt = add_size_class(ann_tgt, bins=bins, labels=size_labels)

        # Load activations
        examples, idx_map = load_examples_for_experiment(exp, args.model, args.categories, allowed_layers=set(args.layers) if args.layers else None)
        if not examples:
            print(f"No examples loaded for {exp}; skipping")
            continue
        if index_map is None:
            index_map = idx_map

        # Extract filenames via the results CSV used by loader
        results_csv = Path(get_experiment_path(exp, "results")) / f"{args.model}_results_Presence.csv"
        if not results_csv.exists():
            candidates = list(Path(get_experiment_path(exp, "results")).glob("*_results_*.csv"))
            if not candidates:
                raise FileNotFoundError(f"No results CSV found for {exp}")
            results_csv = candidates[0]
        df_res = pd.read_csv(results_csv)
        filenames_all = df_res["filename"].astype(str).tolist()

        # Keep only filenames with available activations (to match examples ordering)
        checkpoints_dir = Path(get_experiment_path(exp, "checkpoints"))
        filenames_existing: List[str] = []
        for fn in filenames_all:
            if resolve_activation_path(checkpoints_dir, fn) is not None:
                filenames_existing.append(fn)
        # Build features matrix in the same order
        feats: List[np.ndarray] = [ex["x"].numpy() for ex in examples]
        n = min(len(filenames_existing), len(feats))
        filenames_existing = filenames_existing[:n]
        feats = feats[:n]
        F = np.vstack(feats)

        # Merge with annotations and align by filename using O(n) map
        fname_to_idx = {fn: i for i, fn in enumerate(filenames_existing)}
        merged = pd.merge(pd.DataFrame({"filename": filenames_existing}), ann_tgt, on="filename", how="inner")
        if merged.empty:
            print(f"No target annotations matched filenames for {exp}; skipping")
            continue
        match_idx = [fname_to_idx[fn] for fn in merged["filename"].tolist() if fn in fname_to_idx]
        F = F[match_idx]

        merged["log_n"] = np.log(merged["num_distractors"].astype(float) + 1.0)

        all_rows.append({
            "exp": exp,
            "features": F,
            "meta": merged,
        })

    if not all_rows:
        raise RuntimeError("No data rows collected; ensure experiments and model are correct.")

    # Compute global residualization and global standardization across all experiments
    # 1) Residualize each experiment separately vs its own log_n
    for row in all_rows:
        F = row["features"]
        meta: pd.DataFrame = row["meta"]
        F_resid = residualize_against_log_n(F, meta["log_n"].values.astype(float))
        row["features_resid"] = F_resid

    # 2) Compute global mean/std from concatenated residuals
    concat_resid = np.vstack([row["features_resid"] for row in all_rows if "features_resid" in row])
    g_mean = concat_resid.mean(axis=0, keepdims=True)
    g_std = concat_resid.std(axis=0, keepdims=True)
    g_std[g_std < 1e-6] = 1e-6

    # 3) Standardize each experiment with global stats and pool for axis
    pooled_Fz: List[np.ndarray] = []
    pooled_sizeseries: List[pd.Series] = []
    for row in all_rows:
        Fz = (row["features_resid"] - g_mean) / g_std
        row["features_std"] = Fz
        pooled_Fz.append(Fz)
        pooled_sizeseries.append(row["meta"]["size_class"].astype(str))
    pooled_Fz_arr = np.vstack(pooled_Fz)
    pooled_sizes = pd.concat(pooled_sizeseries, axis=0)

    # 4) Build a single global salience axis from pooled data
    pooled_means = compute_bin_means(pooled_Fz_arr, ["Small", "Medium", "Large"], pooled_sizes)
    # Remove duplicates of label names just in case
    pooled_means = {k: v for k, v in pooled_means.items()}
    w = construct_salience_axis(pooled_means)

    report: Dict[str, Any] = {"experiments": {}, "salience_axis_norm": float(np.linalg.norm(w))}

    # 5) Per-experiment metrics using the shared salience axis
    for row in all_rows:
        exp = row["exp"]
        F = row["features"]
        meta: pd.DataFrame = row["meta"]
        Fz = row["features_std"]
        s = Fz @ w

        mono = meta.assign(score=s).groupby("size_class")["score"].mean().to_dict()
        try:
            present = list(mono.keys())
            pref = {"Small": 0, "Medium": 1, "Large": 2}
            order = sorted(present, key=lambda k: pref.get(str(k), 99))
            vals = [mono[k] for k in order]
            plt.figure(figsize=(4,3))
            plt.plot(order, vals, marker="o")
            plt.title(f"Salience score vs size class: {exp}")
            plt.ylabel("mean w_s^T h~")
            plt.tight_layout()
            out_png = Path("results") / exp / f"{args.model}_salience_monotonicity.png"
            out_png.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_png, dpi=150)
            plt.close()
        except Exception:
            pass

        coords = meta[["center_x", "center_y"]].values.astype(float)
        _, r2_loc = fit_predict_location(Fz, coords)
        Fz_abl = project_out_axis(Fz, w)
        _, r2_loc_abl = fit_predict_location(Fz_abl, coords)

        quad_map = {"Quadrant 1": 0, "Quadrant 2": 1, "Quadrant 3": 2, "Quadrant 4": 3}
        quads = meta["quadrant"].map(quad_map).values
        valid = ~pd.isna(quads)
        quad_acc = None
        quad_acc_abl = None
        if valid.any():
            qlabels = quads[valid].astype(int)
            qF = Fz[valid]
            qF_abl = Fz_abl[valid]
            _, quad_acc = fit_predict_quadrant(qF, qlabels)
            _, quad_acc_abl = fit_predict_quadrant(qF_abl, qlabels)

        report["experiments"][exp] = {
            "num_samples": int(F.shape[0]),
            "feature_dim": int(F.shape[1]),
            "monotonicity_means": mono,
            "location_r2": r2_loc,
            "location_r2_ablated": r2_loc_abl,
            "quadrant_acc": quad_acc,
            "quadrant_acc_ablated": quad_acc_abl,
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


