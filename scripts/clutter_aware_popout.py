import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from rds6_config import get_experiment_path
from scripts.train_on_activations import (
    load_examples_for_experiment,
    resolve_activation_path,
)


def read_annotations(local_results_dir: Path) -> pd.DataFrame:
    csv_path = local_results_dir / "annotations.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"annotations.csv not found at {csv_path}")
    df = pd.read_csv(csv_path)
    required = [
        "filename",
        "target",
        "size",
        "num_distractors",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required annotation columns: {missing}")
    # keep only target rows; size class is defined on targets
    df = df[df["target"] == True].copy()
    return df


def compute_global_size_bins(pooled_sizes: List[float]) -> Tuple[List[float], List[str]]:
    sizes = np.array(pooled_sizes, dtype=float)
    labels = ["Small", "Medium", "Large"]
    uniq = np.unique(sizes)
    if uniq.size >= 3:
        q1, q2 = np.quantile(sizes, [0.33, 0.66])
        if q1 >= q2:
            eps = max(1e-6, 1e-6 * (np.max(sizes) - np.min(sizes) + 1.0))
            q2 = q1 + eps
        bins = [-np.inf, float(q1), float(q2), np.inf]
    elif uniq.size == 2:
        s1, s2 = np.sort(uniq)
        mid = (s1 + s2) / 2.0
        bins = [-np.inf, mid, np.inf]
        labels = ["Small", "Large"]
    else:
        bins = [-np.inf, np.inf]
        labels = ["Medium"]
    return bins, labels


def add_size_class(df: pd.DataFrame, bins: List[float], labels: List[str]) -> pd.DataFrame:
    out = df.copy()
    out["size_class"] = pd.cut(out["size"], bins=bins, labels=labels, include_lowest=True, duplicates="drop")
    return out


def find_results_csv(results_dir: Path, model: str) -> Path:
    candidates = list(results_dir.glob(f"{model}_results_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No results CSV found in {results_dir} for model {model}")
    return candidates[0]


def assemble_features_for_exp(
    exp: str,
    model: str,
    categories: List[str],
    allowed_layers: Optional[Set[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    examples, _ = load_examples_for_experiment(exp, model, categories, allowed_layers=allowed_layers)
    if not examples:
        return np.zeros((0, 0), dtype=np.float32), []

    # Prefer local PWD results; fallback to RDS only if missing
    try:
        results_csv = find_results_csv(Path("results") / exp, model)
    except FileNotFoundError:
        results_csv = find_results_csv(Path(get_experiment_path(exp, "results")), model)
    df_res = pd.read_csv(results_csv)
    filenames_all = df_res["filename"].astype(str).tolist()
    checkpoints_dir = Path(get_experiment_path(exp, "checkpoints"))
    filenames_existing: List[str] = []
    for fn in filenames_all:
        if resolve_activation_path(checkpoints_dir, fn) is not None:
            filenames_existing.append(fn)

    feats: List[np.ndarray] = [ex["x"].numpy() for ex in examples]
    n = min(len(filenames_existing), len(feats))
    return np.vstack(feats[:n]), filenames_existing[:n]


def fit_interaction_probe(
    H: np.ndarray,
    size_class: pd.Series,
    logn: np.ndarray,
) -> Dict[str, Any]:
    # Build feature matrix: [H, (log(n+1))*H]
    z = logn.reshape(-1, 1)
    X = np.hstack([H, H * z])
    # Ordinal classes mapped to 0/1/2, drop rows with NaNs in size_class
    ord_map = {"Small": 0, "Medium": 1, "Large": 2}
    mask = size_class.astype(str).isin(ord_map.keys()).values
    X = X[mask]
    z = z[mask]
    y = size_class[mask].map(ord_map).values.astype(int)
    if X.shape[0] < 5:
        return {"w_base": None, "w_clutter": None, "classes": None}

    # Multinomial logistic regression
    clf = LogisticRegression(max_iter=2000, multi_class="multinomial", solver="lbfgs")
    clf.fit(X, y)
    # Reduce to one-vs-rest weight for ordinal direction via PCA-like combination of class weights.
    # We project the 3-class weights onto a single axis using [-1,0,1] coding.
    W = clf.coef_  # shape [3, 2D]
    coding = np.array([-1.0, 0.0, 1.0]).reshape(3, 1)
    w_ord = (coding.T @ W).reshape(-1)
    D = H.shape[1]
    w_base = w_ord[:D]
    w_clutter = w_ord[D:]
    return {
        "w_base": w_base,
        "w_clutter": w_clutter,
        "classes": ["Small", "Medium", "Large"],
    }


def compute_scores(H: np.ndarray, w_base: np.ndarray, w_clutter: np.ndarray, logn: np.ndarray) -> np.ndarray:
    return H @ (w_base + (logn.reshape(-1, 1) * w_clutter))


def per_condition_slopes(scores: np.ndarray, size_class: pd.Series, logn: np.ndarray) -> Dict[str, float]:
    def slope(x: np.ndarray, y: np.ndarray) -> float:
        X = np.column_stack([np.ones_like(x), x])
        beta = np.linalg.pinv(X.T @ X) @ X.T @ y
        return float(beta[1])
    res: Dict[str, float] = {}
    for cl in ["Small", "Medium", "Large"]:
        idx = (size_class.astype(str) == cl).values
        if np.sum(idx) >= 3:
            res[cl] = slope(logn[idx], scores[idx])
    return res


def auc_pairs(scores: np.ndarray, size_class: pd.Series) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {"Large_vs_Small": None, "Large_vs_Medium": None, "Medium_vs_Small": None}
    series = size_class.astype(str)
    def auc_for(a: str, b: str) -> Optional[float]:
        mask = series.isin([a, b]).values
        y = (series[mask].values == a).astype(int)
        s = scores[mask]
        if s.size == 0 or y.min() == y.max():
            return None
        try:
            return float(roc_auc_score(y, s))
        except Exception:
            return None
    out["Large_vs_Small"] = auc_for("Large", "Small")
    out["Large_vs_Medium"] = auc_for("Large", "Medium")
    out["Medium_vs_Small"] = auc_for("Medium", "Small")
    return out


def angle_change(w_base: np.ndarray, w_clutter: np.ndarray, zbar: float) -> Optional[float]:
    u = w_base
    v = w_base + zbar * w_clutter
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1e-8 or nv < 1e-8:
        return None
    cosang = float(np.clip((u @ v) / (nu * nv + 1e-12), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Clutter-aware pop-out readout analysis")
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--categories", nargs="*", default=["residual_stream"])
    parser.add_argument("--layers", nargs="*", default=None, help="Optional allowed layer keys, e.g., layer_5 layer_50")
    parser.add_argument("--output_json", default="clutter_popout_report.json")
    parser.add_argument("--output_table", default="table_s1.csv")
    args = parser.parse_args()

    # 1) Determine global bins for size classes
    pooled_sizes: List[float] = []
    exp_to_ann: Dict[str, pd.DataFrame] = {}
    for exp in args.experiments:
        ann = read_annotations(Path("results") / exp)
        exp_to_ann[exp] = ann
        pooled_sizes.extend(ann["size"].astype(float).tolist())
    if not pooled_sizes:
        raise RuntimeError("No target rows with size found in annotations across experiments")
    bins, labels = compute_global_size_bins(pooled_sizes)

    # 2) Per experiment: assemble features, align with annotations, compute logn
    rows: List[Dict[str, Any]] = []
    for exp in args.experiments:
        H, filenames = assemble_features_for_exp(exp, args.model, args.categories, allowed_layers=set(args.layers) if args.layers else None)
        if H.shape[0] == 0:
            continue
        ann = add_size_class(exp_to_ann[exp], bins, labels)
        # Align by filename
        fname_to_idx = {fn: i for i, fn in enumerate(filenames)}
        ann = ann[ann["filename"].isin(fname_to_idx.keys())].copy()
        if ann.empty:
            continue
        idx = [fname_to_idx[fn] for fn in ann["filename"].tolist()]
        H = H[idx]
        logn = np.log(ann["num_distractors"].astype(float).values + 1.0)
        size_cls = ann["size_class"].astype(str)
        rows.append({"exp": exp, "H": H, "logn": logn, "size": size_cls})

    if not rows:
        raise RuntimeError("No aligned features/annotations found; check experiments and model")

    # 3) Determine representative early/mid/late layers available from saved activations (if provided), else treat concatenated H as a single layer
    # Here we fit a single probe per requested layer set; if multiple layers are provided via --layers, we will report per provided key in the output JSON by refitting on masked features is future work.

    # 4) Fit interaction probe on pooled data to get shared direction per layer set
    H_pool = np.vstack([r["H"] for r in rows])
    logn_pool = np.concatenate([r["logn"] for r in rows])
    size_pool = pd.concat([r["size"] for r in rows], axis=0)

    fit = fit_interaction_probe(H_pool, size_pool, logn_pool)
    w_base = fit["w_base"]
    w_clutter = fit["w_clutter"]
    if w_base is None or w_clutter is None:
        raise RuntimeError("Failed to fit interaction probe; not enough data")

    # 5) Per experiment: compute scores, slopes, and AUCs; plot Figure 1 panel per representative layer set
    report: Dict[str, Any] = {"experiments": {}, "labels": labels}
    table_rows: List[Dict[str, Any]] = []

    for r in rows:
        exp = r["exp"]
        H = r["H"]
        logn = r["logn"]
        size_cls = r["size"]
        s = compute_scores(H, w_base, w_clutter, logn)
        slopes = per_condition_slopes(s, size_cls, logn)
        aucs = auc_pairs(s, size_cls)
        zbar = float(np.mean(logn))
        ang = angle_change(w_base, w_clutter, zbar)

        # Figure 1: pop-out score vs distractor count by size class
        try:
            df_plot = pd.DataFrame({"score": s, "logn": logn, "size": size_cls.values})
            fig, ax = plt.subplots(figsize=(4.5, 3.2))
            for cl, color in zip(["Small", "Medium", "Large"], ["tab:blue", "tab:orange", "tab:green"]):
                sub = df_plot[df_plot["size"] == cl]
                if not sub.empty:
                    ax.scatter(sub["logn"], sub["score"], s=8, alpha=0.4, color=color, label=cl)
                    # trend line
                    X = np.column_stack([np.ones(len(sub)), sub["logn"].values])
                    beta = np.linalg.pinv(X.T @ X) @ X.T @ sub["score"].values
                    xs = np.linspace(sub["logn"].min(), sub["logn"].max(), 50)
                    ys = beta[0] + beta[1] * xs
                    ax.plot(xs, ys, color=color)
            ax.set_title(f"{exp}: pop-out vs log n")
            ax.set_xlabel("log(n+1)")
            ax.set_ylabel("s = h^T(w_base + log(n+1) w_clutter)")
            ax.legend(frameon=False)
            out_png = Path("results") / exp / f"{args.model}_clutter_popout.png"
            fig.tight_layout()
            fig.savefig(out_png, dpi=150)
            plt.close(fig)
        except Exception:
            pass

        report["experiments"][exp] = {
            "slopes": slopes,
            "aucs": aucs,
            "angle_deg": ang,
            "num_samples": int(H.shape[0]),
            "feature_dim": int(H.shape[1]),
        }

        table_rows.append({
            "experiment": exp,
            "auc_large_vs_small": aucs.get("Large_vs_Small"),
            "auc_large_vs_medium": aucs.get("Large_vs_Medium"),
            "auc_medium_vs_small": aucs.get("Medium_vs_Small"),
            "slope_small": slopes.get("Small"),
            "slope_medium": slopes.get("Medium"),
            "slope_large": slopes.get("Large"),
            "angle_deg": ang,
        })

    # Save JSON report and Table S1 CSV
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))

    out_table = Path(args.output_table)
    pd.DataFrame(table_rows).to_csv(out_table, index=False)
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_table}")


if __name__ == "__main__":
    main()


