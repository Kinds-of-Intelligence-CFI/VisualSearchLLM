#!/usr/bin/env python3
"""
Test explicit hypotheses against probe battery results.

Inputs:
- Battery summary CSV written by scripts/train_on_activations.py (models/JOINT/<model>_battery_summary.csv)

Hypotheses:
- H1 (Circles, early layers): CircleSizesLarge > CircleSizesMedium > CircleSizesSmall in early layers.
- H2 (2Among5): ColourRand > NoColourRand > ConjRand, reflected in activations (combined and late layers).
- H3 (LitSpheres): Top/Bottom "pop out" more than Left/Right, especially at late layers, and show stronger late-early gains.

Outputs:
- JSON report with PASS/WEAK/FAIL per hypothesis, effect sizes, and supporting stats.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from rds6_config import get_model_path


def load_summary(model: str, summary_path: Optional[str]) -> pd.DataFrame:
    if summary_path:
        path = Path(summary_path)
    else:
        path = Path(get_model_path("JOINT", f"{model}_battery_summary.csv"))
    if not path.exists():
        raise FileNotFoundError(f"Battery summary not found: {path}")
    df = pd.read_csv(path)
    # Normalize identifier to string
    df["identifier"] = df["identifier"].astype(str)
    return df


def is_per_layer(row: pd.Series) -> bool:
    return row["test"] == "per_layer"


def parse_layer_index(identifier: str) -> Optional[int]:
    if identifier.startswith("layer_"):
        try:
            return int(identifier.split("_")[-1])
        except Exception:
            return None
    return None


def split_layer_bins(indices: List[int]) -> Tuple[List[int], List[int], List[int]]:
    if not indices:
        return [], [], []
    uniq = sorted({int(i) for i in indices})
    # Percentile-based bins
    q1 = np.percentile(uniq, 33)
    q2 = np.percentile(uniq, 66)
    early = [int(i) for i in uniq if i <= q1]
    mid = [int(i) for i in uniq if q1 < i <= q2]
    late = [int(i) for i in uniq if i > q2]
    # Fallbacks to avoid empty bins when very sparse
    if not early and uniq:
        early = [int(uniq[0])]
    if not late and uniq:
        late = [int(uniq[-1])]
    return early, mid, late


def mean_acc_for_layers(df: pd.DataFrame, exp: str, layer_idxs: List[int]) -> float:
    if not layer_idxs:
        return float("nan")
    rows = []
    for i in layer_idxs:
        ident = f"layer_{int(i)}"
        m = df[(df["experiment"] == exp) & (df["test"] == "per_layer") & (df["identifier"] == ident)]
        if len(m) == 1:
            rows.append(float(m.iloc[0]["val_accuracy"]))
    if not rows:
        return float("nan")
    return float(np.mean(rows))


def best_acc_for_layers(df: pd.DataFrame, exp: str, layer_idxs: List[int]) -> float:
    if not layer_idxs:
        return float("nan")
    vals = []
    for i in layer_idxs:
        ident = f"layer_{int(i)}"
        m = df[(df["experiment"] == exp) & (df["test"] == "per_layer") & (df["identifier"] == ident)]
        if len(m) == 1:
            vals.append(float(m.iloc[0]["val_accuracy"]))
    if not vals:
        return float("nan")
    return float(np.max(vals))


def first_last_layer_acc(df: pd.DataFrame, exp: str) -> Tuple[Optional[float], Optional[float]]:
    per = df[(df["experiment"] == exp) & (df["test"] == "per_layer")]
    if per.empty:
        return None, None
    per = per.copy()
    per["layer_idx"] = per["identifier"].apply(parse_layer_index)
    per = per.dropna(subset=["layer_idx"])
    if per.empty:
        return None, None
    first_row = per.sort_values("layer_idx").iloc[0]
    last_row = per.sort_values("layer_idx").iloc[-1]
    return float(first_row["val_accuracy"]), float(last_row["val_accuracy"])


def per_layer_map(df: pd.DataFrame, exp: str) -> Dict[int, float]:
    per = df[(df["experiment"] == exp) & (df["test"] == "per_layer")].copy()
    if per.empty:
        return {}
    per["layer_idx"] = per["identifier"].apply(parse_layer_index)
    per = per.dropna(subset=["layer_idx"])  # drop input/output rows
    out: Dict[int, float] = {}
    for _, row in per.iterrows():
        out[int(row["layer_idx"])] = float(row["val_accuracy"])  # last value wins if duplicate
    return out


def test_h1(df: pd.DataFrame) -> Dict[str, Any]:
    # Gather available layer indices across any circle experiment
    circle_exps = ["CircleSizesSmall", "CircleSizesMedium", "CircleSizesLarge"]
    per = df[(df["experiment"].isin(circle_exps)) & (df["test"] == "per_layer")]
    per = per.copy()
    per["layer_idx"] = per["identifier"].apply(parse_layer_index)
    per = per.dropna(subset=["layer_idx"])
    indices = per["layer_idx"].tolist()
    early, mid, late = split_layer_bins(indices)

    # Early-layer mean accuracies
    acc_small = mean_acc_for_layers(df, "CircleSizesSmall", early)
    acc_med = mean_acc_for_layers(df, "CircleSizesMedium", early)
    acc_large = mean_acc_for_layers(df, "CircleSizesLarge", early)

    ordering_ok = (acc_large > acc_med) and (acc_med > acc_small)
    margin = min(acc_large - acc_med, acc_med - acc_small) if all(np.isfinite([acc_small, acc_med, acc_large])) else float("nan")
    status = "FAIL"
    if ordering_ok and margin >= 0.05:
        status = "PASS"
    elif ordering_ok:
        status = "WEAK"

    data_used = {
        "early_layers": [int(i) for i in early],
        "per_experiment": {
            "CircleSizesSmall": {i: mean_acc_for_layers(df, "CircleSizesSmall", [i]) for i in early},
            "CircleSizesMedium": {i: mean_acc_for_layers(df, "CircleSizesMedium", [i]) for i in early},
            "CircleSizesLarge": {i: mean_acc_for_layers(df, "CircleSizesLarge", [i]) for i in early},
        },
    }

    return {
        "id": "H1",
        "description": "Circles: Large > Medium > Small in early layers",
        "status": status,
        "early_layers": [int(i) for i in early],
        "acc_small": acc_small,
        "acc_medium": acc_med,
        "acc_large": acc_large,
        "min_pairwise_margin": margin,
        "data_used": data_used,
    }


def test_h2(df: pd.DataFrame) -> Dict[str, Any]:
    exps = ["2Among5ColourRand", "2Among5NoColourRand", "2Among5ConjRand"]
    # Combined probe comparison
    combined: Dict[str, float] = {}
    for e in exps:
        m = df[(df["experiment"] == e) & (df["test"] == "combined")]
        combined[e] = float(m.iloc[0]["val_accuracy"]) if len(m) == 1 else float("nan")

    # Late-layer comparison (max across late bin)
    per = df[(df["test"] == "per_layer") & (df["experiment"].isin(exps))].copy()
    per["layer_idx"] = per["identifier"].apply(parse_layer_index)
    per = per.dropna(subset=["layer_idx"])
    early, mid, late = split_layer_bins(per["layer_idx"].tolist())

    late_best: Dict[str, float] = {}
    for e in exps:
        late_best[e] = best_acc_for_layers(df, e, late)

    # Desired ordering: ColourRand > NoColourRand > ConjRand
    c, n, j = combined.get(exps[0], np.nan), combined.get(exps[1], np.nan), combined.get(exps[2], np.nan)
    lc, ln, lj = late_best.get(exps[0], np.nan), late_best.get(exps[1], np.nan), late_best.get(exps[2], np.nan)

    ordering_combined = (c > n) and (n > j)
    ordering_late = (lc > ln) and (ln > lj)

    margin_combined = min(c - n, n - j) if all(np.isfinite([c, n, j])) else float("nan")
    margin_late = min(lc - ln, ln - lj) if all(np.isfinite([lc, ln, lj])) else float("nan")

    status = "FAIL"
    if ordering_combined and ordering_late and min(margin_combined, margin_late) >= 0.05:
        status = "PASS"
    elif ordering_combined or ordering_late:
        status = "WEAK"

    data_used = {
        "combined": combined,
        "late_layers": [int(i) for i in late],
        "per_experiment_late": {
            e: {i: mean_acc_for_layers(df, e, [i]) for i in late} for e in exps
        },
    }

    return {
        "id": "H2",
        "description": "2Among5: ColourRand > NoColourRand > ConjRand (combined and late layers)",
        "status": status,
        "combined_acc": combined,
        "late_best_acc": late_best,
        "margin_combined": margin_combined,
        "margin_late": margin_late,
        "late_layers": [int(i) for i in late],
        "data_used": data_used,
    }


def test_h3(df: pd.DataFrame) -> Dict[str, Any]:
    tops = ["LitSpheresTop", "LitSpheresBottom"]
    sides = ["LitSpheresLeft", "LitSpheresRight"]
    per = df[(df["test"] == "per_layer") & (df["experiment"].isin(tops + sides))].copy()
    if per.empty:
        return {"id": "H3", "status": "FAIL", "reason": "No per-layer rows"}
    per["layer_idx"] = per["identifier"].apply(parse_layer_index)
    per = per.dropna(subset=["layer_idx"])
    early, mid, late = split_layer_bins(per["layer_idx"].tolist())

    # Late accuracy and late-early gain for each group
    def metrics_for(exp: str) -> Dict[str, float]:
        late_acc = best_acc_for_layers(df, exp, late)
        first_acc, last_acc = first_last_layer_acc(df, exp)
        gain = None
        if first_acc is not None and last_acc is not None:
            gain = float(last_acc - first_acc)
        return {"late_best": late_acc, "gain_late_minus_early": gain}

    m_top = {e: metrics_for(e) for e in tops}
    m_side = {e: metrics_for(e) for e in sides}

    # Averages
    avg_top_late = float(np.nanmean([m_top[e]["late_best"] for e in tops]))
    avg_side_late = float(np.nanmean([m_side[e]["late_best"] for e in sides]))
    avg_top_gain = float(np.nanmean([m_top[e]["gain_late_minus_early"] for e in tops]))
    avg_side_gain = float(np.nanmean([m_side[e]["gain_late_minus_early"] for e in sides]))

    status = "FAIL"
    cond1 = avg_top_late > avg_side_late
    cond2 = avg_top_gain > avg_side_gain
    margin1 = avg_top_late - avg_side_late
    margin2 = avg_top_gain - avg_side_gain
    if cond1 and cond2 and min(margin1, margin2) >= 0.05:
        status = "PASS"
    elif cond1 or cond2:
        status = "WEAK"

    data_used = {
        "late_layers": [int(i) for i in late],
        "per_experiment_per_layer": {
            e: per_layer_map(df, e) for e in tops + sides
        },
        "first_last": {
            e: {
                "first_acc": first_last_layer_acc(df, e)[0],
                "last_acc": first_last_layer_acc(df, e)[1],
            } for e in tops + sides
        },
    }

    return {
        "id": "H3",
        "description": "LitSpheres: Top/Bottom pop out more than Left/Right (late and gain)",
        "status": status,
        "late_layers": [int(i) for i in late],
        "avg_top_late": avg_top_late,
        "avg_side_late": avg_side_late,
        "avg_top_gain": avg_top_gain,
        "avg_side_gain": avg_side_gain,
        "margin_late": margin1,
        "margin_gain": margin2,
        "per_exp_top": m_top,
        "per_exp_side": m_side,
        "data_used": data_used,
    }


def _to_json_safe(obj: Any) -> Any:
    # Convert NaN/Inf to None, numpy types to Python types, and ensure ints remain ints
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        if not np.isfinite(x):
            return None
        return x
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float):
        if not np.isfinite(obj):
            return None
        return obj
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["llama11B", "llama90B"], default="llama11B")
    parser.add_argument("--summary", help="Path to battery summary CSV (optional)")
    parser.add_argument("--out", help="Path to write hypothesis report JSON (optional)")
    args = parser.parse_args()

    df = load_summary(args.model, args.summary)

    h1 = test_h1(df)
    h2 = test_h2(df)
    h3 = test_h3(df)

    report = {
        "model": args.model,
        "hypotheses": [h1, h2, h3],
    }

    out_path = Path(args.out) if args.out else Path(get_model_path("JOINT", f"{args.model}_hypothesis_report.json"))
    report_safe = _to_json_safe(report)
    out_path.write_text(json.dumps(report_safe, indent=2))
    print(f"✓ Wrote hypothesis report: {out_path}")
    # Console summary
    for h in report["hypotheses"]:
        print(f"{h['id']}: {h['status']} — {h['description']}")


if __name__ == "__main__":
    main()


