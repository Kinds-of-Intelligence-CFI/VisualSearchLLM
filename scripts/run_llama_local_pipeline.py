import argparse
import os
import subprocess
from pathlib import Path


EXPERIMENT_PRESETS = {
    # 2Among5 (Cells)
    "2Among5ColourRand": "2Among5ColourRand",
    "2Among5NoColourRand": "2Among5NoColourRand",
    "2Among5ConjRand": "2Among5ConjRand",
    # Light Priors (Cells)
    "LitSpheresTop": "LitSpheresTop",
    "LitSpheresBottom": "LitSpheresBottom",
    "LitSpheresLeft": "LitSpheresLeft",
    "LitSpheresRight": "LitSpheresRight",
    # Circle Sizes (Cells)
    "CircleSizesSmall": "CircleSizesSmall",
    "CircleSizesMedium": "CircleSizesMedium",
    "CircleSizesLarge": "CircleSizesLarge",
}


PROMPT_BY_EXPERIMENT = {
    # 2Among5 prompts (cells)
    "2Among5ColourRand": "std2x2-2Among5",
    "2Among5NoColourRand": "std2x2-2Among5",
    "2Among5ConjRand": "std2x2-2Among5-conj",
    # Light Priors prompts (cells)
    "LitSpheresTop": "LightPriorsOOO",
    "LitSpheresBottom": "LightPriorsOOO",
    "LitSpheresLeft": "LightPriorsOOO",
    "LitSpheresRight": "LightPriorsOOO",
    # Circle Sizes prompt (cells)
    "CircleSizesSmall": "circle-sizes",
    "CircleSizesMedium": "circle-sizes",
    "CircleSizesLarge": "circle-sizes",
}

def run(cmd, cwd) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_results_dir() -> None:
    Path("results").mkdir(exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["llama11B", "llama90B"], default="llama11B")
    parser.add_argument("--num", type=int, default=150, help="Images per experiment directory")
    parser.add_argument("--experiments", nargs="*", default=list(EXPERIMENT_PRESETS.keys()),
                        help="Subset of experiment directory names to run")
    parser.add_argument("--python", default="python", help="Python executable to use")
    args = parser.parse_args()

    ensure_results_dir()

    # 1) Generate images
    for exp_dir in args.experiments:
        if exp_dir not in EXPERIMENT_PRESETS:
            raise ValueError(f"Unknown experiment directory: {exp_dir}")
        preset = EXPERIMENT_PRESETS[exp_dir]
        run([args.python, "generateImages.py", "-n", str(args.num), "-d", exp_dir, "-p", preset], cwd=str(Path(".").resolve()))

    # 2) Create batches named for the selected model so submitHPCBatch can find them
    for exp_dir in args.experiments:
        prompt = PROMPT_BY_EXPERIMENT[exp_dir]
        run([args.python, "createBatch.py", "-d", exp_dir, "-m", "llamaLocal", "-p", prompt], cwd=str(Path(".").resolve()))

    # 3) Submit to SLURM (asynchronous). One job per batch file.
    for exp_dir in args.experiments:
        run([args.python, "submitHPCBatch.py", "-d", exp_dir, "-m", args.model], cwd=str(Path(".").resolve()))

    # 4) Print follow-up commands to combine and process once jobs finish
    print("\n🚀 SLURM submissions complete. When jobs finish, combine and process results with:")
    for exp_dir in args.experiments:
        print(f"  python {Path('combineLlama.py')} -m {args.model} -d {exp_dir}")
        print(f"  python {Path('processBatchResults.py')} -d {exp_dir} -rc -m {args.model} -b combined_batch_responses.jsonl")
    print("\nℹ️ Jobs are asynchronous. Monitor with: squeue -u $USER and tail -f logs/llama_<jobid>.out")


if __name__ == "__main__":
    main()


