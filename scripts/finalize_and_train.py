import argparse
import subprocess
from pathlib import Path

EXPERIMENTS = [
    "2Among5ColourRand", "2Among5NoColourRand", "2Among5ConjRand",
    "LitSpheresTop", "LitSpheresBottom", "LitSpheresRight", "LitSpheresLeft",
    "CircleSizesSmall", "CircleSizesMedium", "CircleSizesLarge",
]

def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def finalize(model: str, experiments: list[str], py: str) -> None:
    for exp in experiments:
        # Combine
        run([py, "combineLlama.py", "-m", model, "-d", exp])
        # Process (Cells mode)
        run([py, "processBatchResults.py", "-d", exp, "-rc", "-m", model, "-b", "combined_batch_responses.jsonl"])
        print(f"[{exp}] ✅ combined + processed")


def train(model: str, py: str) -> None:
    run([py, "scripts/train_on_activations.py", "--model", model])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["llama11B", "llama90B"], default="llama11B")
    parser.add_argument("--experiments", nargs="*", default=EXPERIMENTS)
    parser.add_argument("--python", default="python3")
    args = parser.parse_args()

    # Ensure we are at repo root
    root = Path(__file__).resolve().parents[1]
    print(f"Repo root: {root}")

    finalize(args.model, args.experiments, args.python)
    '''train(args.model, args.python)'''
    print("\n🎉 Finalized all experiments and trained on activations.")


if __name__ == "__main__":
    main()


