#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import glob
import math
import torch
import numpy as np


def find_default_file(basename: str) -> Path | None:
    candidates = sorted(glob.glob(f"results/*/checkpoints/{basename}"))
    if not candidates:
        return None
    return Path(candidates[0]).resolve()


def describe_array(name: str, arr) -> str:
    try:
        if isinstance(arr, torch.Tensor):
            shape = tuple(arr.shape)
            dtype = str(arr.dtype)
            device = str(arr.device)
            return f"{name}: tensor shape={shape} dtype={dtype} device={device}"
        else:
            np_arr = np.asarray(arr)
            shape = tuple(np_arr.shape)
            dtype = str(np_arr.dtype)
            return f"{name}: array  shape={shape} dtype={dtype}"
    except Exception as e:
        return f"{name}: <unprintable> ({type(arr).__name__}): {e}"


def print_stats(arr) -> str:
    try:
        if isinstance(arr, torch.Tensor):
            a = arr.detach().to("cpu", dtype=torch.float32)
            vmin = float(a.min())
            vmax = float(a.max())
            mean = float(a.mean())
            std = float(a.std(unbiased=False))
        else:
            a = np.asarray(arr, dtype=np.float32)
            if a.size == 0 or np.any(~np.isfinite(a)):
                return "(stats unavailable)"
            vmin = float(np.min(a))
            vmax = float(np.max(a))
            mean = float(np.mean(a))
            std = float(np.std(a))
        # Avoid printing NaN/Inf
        def fmt(x: float) -> str:
            if not math.isfinite(x):
                return "nan"
            return f"{x:.6g}"
        return f"min={fmt(vmin)} max={fmt(vmax)} mean={fmt(mean)} std={fmt(std)}"
    except Exception:
        return "(stats error)"


def inspect_file(path: Path, show_stats: bool) -> None:
    print(f"Loading: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    print(f"Top-level type: {type(data).__name__}")

    if not isinstance(data, dict):
        print("Not a dict — raw print follows:\n", repr(data))
        return

    categories = list(data.keys())
    print(f"Categories: {categories}")

    for category, layers in data.items():
        if not isinstance(layers, dict):
            print(f"- {category}: {type(layers).__name__}")
            continue
        print(f"- {category}: {len(layers)} entries")
        for layer_name in sorted(layers.keys()):
            arr = layers[layer_name]
            line = describe_array(f"    {layer_name}", arr)
            if show_stats and arr is not None:
                line += "  " + print_stats(arr)
            print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect saved activation file contents")
    parser.add_argument("file", nargs="?", help="Path to *_activations.pt (defaults to first match under results/*/checkpoints)")
    parser.add_argument("--stats", action="store_true", help="Print min/max/mean/std per entry")
    args = parser.parse_args()

    if args.file:
        path = Path(args.file).expanduser().resolve()
    else:
        default = find_default_file("image_91.png_activations.pt")
        if default is None:
            print("Could not find image_91.png_activations.pt under results/*/checkpoints. Pass a path explicitly.")
            return
        path = default

    if not path.exists():
        print(f"File not found: {path}")
        return

    inspect_file(path, args.stats)


if __name__ == "__main__":
    main()


