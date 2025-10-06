import pandas as pd
import os
from pathlib import Path

path = os.getcwd()
results_path = os.path.join(path, "ai_results")
base_dir = Path(os.path.join(results_path, "resultsCSVsLightPriors"))

all_dfs = []

for subfolder in ["Top", "Bottom", "Left", "Right"]:
    folder = base_dir / subfolder
    for file in folder.glob("*.csv"):
        # Load csv
        df = pd.read_csv(file)

        # Extract info
        label = subfolder
        source_file = file.name
        source_dir = file.stem.split("_results_")[0]  # e.g. HorizontalGradientSeedNPDet
        model = source_file.split("_results_")[0]  # e.g. claude-haiku

        # Add metadata columns
        df.insert(7, "label", label)
        df.insert(8, "model", model)
        df.insert(9, "source_dir", source_dir)
        df.insert(10, "source_file", source_file)

        all_dfs.append(df)

# Concatenate everything
final_df = pd.concat(all_dfs, ignore_index=True)

# Save to one big CSV
final_df.to_csv(os.path.join(results_path, "e4_light_priors_2_ai.csv"), index=False)