import os
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

# Load data
path = os.getcwd()
results_path = os.path.join(path, "ai_results")
experiments = ["e1_numbers_ai", "e2_light_priors_ai", "e3_circle_sizes_ai"]
experiment = experiments[2]
df = pd.read_csv(os.path.join(results_path, f"{experiment}.csv"))
df["condition"] = df["label"]
df["accuracy"] = df["correct"].astype(int)
conditions = df['condition'].unique()
ai_models = df['model'].unique()

# Run correlations
results = []
raw_pvals = []
for ai_model in ai_models:
    for condition in conditions:
        subset = df[(df['model'] == ai_model) & (df['condition'] == condition)]
        if len(subset) < 2:
            r, p = float('nan'), float('nan')  # Not enough data
        else:
            r, p = pearsonr(subset['num_distractors'], subset['accuracy'])
        results.append({
            'Model': ai_model,
            'Condition': condition,
            'r': r,
            'raw_p': p
        })
        raw_pvals.append(p)

# Adjust p-values
_, adj_pvals, _, _ = multipletests(raw_pvals, method='bonferroni')
for i, adj_p in enumerate(adj_pvals):
    results[i]['p'] = "< .001" if adj_p < 0.001 else f"{adj_p:.3f}"
for r in results:
    del r['raw_p']

# Convert to DataFrame
df_results = pd.DataFrame(results)
pivoted = df_results.pivot(index='Model', columns='Condition')[['r', 'p']]
pivoted.columns = [f"{col[1]}_{col[0]}" for col in pivoted.columns]
pivoted = pivoted.reset_index()

# Convert to Latex table
latex_lines = []
header_conditions = ' & '.join([f"\\multicolumn{{2}}{{c}}{{\\textbf{{{cond}}}}}" for cond in conditions])
sub_headers = ' & '.join(['$r$ & $p$' for _ in conditions])

latex_lines.append("\\begin{table}[h]")
latex_lines.append("\\centering")
latex_lines.append("\\begin{tabular}{" + "l|" + "cc|" * (len(conditions) - 1) + "cc}")
latex_lines.append("\\toprule")
latex_lines.append(f"& {header_conditions} \\\\")
latex_lines.append("\\textbf{Model} & " + sub_headers + " \\\\")
latex_lines.append("\\midrule")

for _, row in pivoted.iterrows():
    row_str = f"{row['Model']}"
    for condition in conditions:
        r = row.get(f"{condition}_r", 'NA')
        p = row.get(f"{condition}_p", 'NA')
        if isinstance(r, float):
            r = f"{r:.3f}"
        row_str += f" & {r} & {p}"
    row_str += " \\\\"
    latex_lines.append(row_str)

latex_lines.append("\\bottomrule")
latex_lines.append("\\end{tabular}")
latex_lines.append("\\caption{Correlation ($r$) between number of distractors and accuracy across AI models and conditions.}")
latex_lines.append("\\label{tab:correlations_circle_sizes}")
latex_lines.append("\\end{table}")

# Print LaTeX
latex_code = '\n'.join(latex_lines)
print(latex_code)

# print descriptives
# Compute means per model and condition
mean_df = df.groupby(['model', 'condition'])['accuracy'].mean().reset_index()
mean_df.rename(columns={'accuracy': 'mean_accuracy'}, inplace=True)

# Print means in readable format
print("\nMean accuracies by model and condition:\n")
for model in ai_models:
    print(f"Model: {model}")
    for condition in conditions:
        mean = mean_df[(mean_df['model'] == model) & (mean_df['condition'] == condition)]['mean_accuracy']
        mean_val = mean.values[0] if not mean.empty else 'NA'
        if isinstance(mean_val, float):
            mean_val = f"{mean_val:.3f}"
        print(f"  {condition}: {mean_val}")
    print()


