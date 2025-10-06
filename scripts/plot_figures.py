#!/usr/bin/env python3
"""
Generate publication-ready plots from probe results for ICLR.

Figures produced:
- Per-layer validation accuracy curves per experiment (PDF)
- Attribution top-15 (category,layer) contribution bars per experiment (PDF)
- Zero-out sensitivity bar (before vs after) per experiment, if available (PDF)
- Hypotheses effect size bar chart (margins) (PDF)

Inputs:
- Battery summary CSV: models/JOINT/<model>_battery_summary.csv
- Per-experiment attribution JSON: models/<EXP>/<model>_probe_attribution.json
- Hypothesis report JSON: models/JOINT/<model>_hypothesis_report.json
"""

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from rds6_config import get_model_path


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_figure(fig, out_base: Path, fmt: str) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    if fmt in ("png", "both"):
        fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    if fmt in ("pdf", "both"):
        fig.savefig(out_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight", facecolor="white")


def layer_idx(identifier: str):
    if isinstance(identifier, str) and identifier.startswith("layer_"):
        try:
            return int(identifier.split("_")[1])
        except Exception:
            return None
    return None


def plot_per_layer_curves(model: str, experiments: List[str], outdir: Path, summary_csv: Path, fmt: str) -> None:
    sns.set_theme(style="whitegrid")
    df = pd.read_csv(summary_csv)
    for e in experiments:
        d = df[(df.experiment == e) & (df.test == "per_layer")].copy()
        d["layer_idx"] = d.identifier.apply(layer_idx)
        d = d.dropna(subset=["layer_idx"]).sort_values("layer_idx")
        if d.empty:
            continue
        fig = plt.figure(figsize=(3.5, 2.5), dpi=300)
        sns.lineplot(data=d, x="layer_idx", y="val_accuracy", marker="o", linewidth=1, color="#4c78a8")
        plt.xlabel("Layer"); plt.ylabel("Val accuracy"); plt.ylim(0.45, 1.0); plt.title(e)
        plt.tight_layout(); save_figure(fig, outdir / f"{e}_per_layer_accuracy", fmt); plt.close()


def plot_attribution_bars(model: str, experiments: List[str], outdir: Path, fmt: str) -> None:
    sns.set_theme(style="whitegrid")
    for e in experiments:
        path = Path(get_model_path(e, f"{model}_probe_attribution.json"))
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        rows = d.get("avg_group_contributions", [])
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: abs(r.get("avg_contribution", 0.0)), reverse=True)[:15]
        labels = [f"{r['category']}:{r['layer']}" for r in rows]
        vals = [r.get("avg_contribution", 0.0) for r in rows]
        fig = plt.figure(figsize=(4.2, 2.8), dpi=300)
        sns.barplot(x=vals, y=labels, orient="h", palette="vlag")
        plt.xlabel("Avg contribution (w·x)"); plt.ylabel("(category:layer)")
        plt.tight_layout(); save_figure(fig, outdir / f"{e}_attr_top15", fmt); plt.close()


def plot_zeroout(model: str, experiments: List[str], outdir: Path, fmt: str) -> None:
    sns.set_theme(style="whitegrid")
    for e in experiments:
        path = Path(get_model_path(e, f"{model}_probe_attribution.json"))
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        if "zero_topk" not in d:
            continue
        before = d.get("logits_before_mean", None)
        after = d.get("logits_after_mean", None)
        if before is None or after is None:
            continue
        fig = plt.figure(figsize=(3.3, 2.3), dpi=300)
        sns.barplot(x=["before", "after"], y=[before, after], palette=["#4c78a8", "#e45756"])
        plt.ylabel("Mean probe logit"); plt.title(f"{e}: zero-out top-{d['zero_topk']}")
        plt.tight_layout(); save_figure(fig, outdir / f"{e}_zeroout_bar", fmt); plt.close()


def plot_hypotheses_margins(model: str, outdir: Path, hyp_json: Path, fmt: str) -> None:
    sns.set_theme(style="whitegrid")
    rep = json.loads(hyp_json.read_text())
    H = {h["id"]: h for h in rep.get("hypotheses", [])}
    labels = ["H1 margin", "H2 combined", "H2 late", "H3 late", "H3 gain"]
    vals = [
        H.get("H1", {}).get("min_pairwise_margin") or 0.0,
        H.get("H2", {}).get("margin_combined") or 0.0,
        H.get("H2", {}).get("margin_late") or 0.0,
        H.get("H3", {}).get("margin_late") or 0.0,
        H.get("H3", {}).get("margin_gain") or 0.0,
    ]
    fig = plt.figure(figsize=(4.0, 2.5), dpi=300)
    sns.barplot(x=labels, y=vals, color="#4c78a8")
    plt.axhline(0.05, color="gray", linestyle="--", linewidth=1)
    plt.xticks(rotation=25, ha="right"); plt.ylabel("Effect size (margin)")
    plt.tight_layout(); save_figure(fig, outdir / "hypotheses_margins", fmt); plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["llama11B", "llama90B"], default="llama90B")
    parser.add_argument("--experiments", nargs="*",
                        default=[
                            "2Among5 Disjunctive", "2Among5 Shape Conjunctive", "2Among5 Shape-Colour Conjunctive",
                            "Light Priors Top", "Light Priors Bottom", "Light Priors Left", "Light Priors Right",
                            "Circle Sizes Small", "Circle Sizes Medium", "Circle Sizes Large",
                        ])
    parser.add_argument("--outdir", help="Directory to write plots (defaults to RDS6 models/JOINT/plots)")
    parser.add_argument("--format", choices=["png", "pdf", "both"], default="png", help="Output format for figures")
    args = parser.parse_args()

    model = args.model
    outdir = Path(args.outdir) if args.outdir else Path(get_model_path("JOINT", "plots"))
    ensure_outdir(outdir)

    summary_csv = Path(get_model_path("JOINT", f"{model}_battery_summary.csv"))
    hyp_json = Path(get_model_path("JOINT", f"{model}_hypothesis_report.json"))

    plot_per_layer_curves(model, args.experiments, outdir, summary_csv, args.format)
    plot_attribution_bars(model, args.experiments, outdir, args.format)
    plot_zeroout(model, args.experiments, outdir, args.format)
    if hyp_json.exists():
        plot_hypotheses_margins(model, outdir, hyp_json, args.format)
    print(f"✓ Wrote plots to: {outdir}")


if __name__ == "__main__":
    main()


