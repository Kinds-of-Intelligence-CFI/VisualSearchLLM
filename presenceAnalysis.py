import os
import pandas as pd
import numpy as np
import argparse
import re
import math
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import textwrap

# Parse command-line arguments for multiple directories and labels
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directories", nargs="+", required=True,
                    help="List of directories containing results CSV files (Presence mode)")
parser.add_argument("-l", "--labels", nargs="+", required=True,
                    help="List of labels for each directory.")
args = parser.parse_args()

if len(args.directories) != len(args.labels):
    raise ValueError("The number of directories and labels must be the same.")

# If only one directory is given, enable figure saving to that directory.
if len(args.directories) == 1:
    save_figures = True
    output_dir = os.path.join("results", args.directories[0])
else:
    save_figures = False
    output_dir = None

# Map each directory to its corresponding label
dir_label_map = dict(zip(args.directories, args.labels))

# Read in presence mode results from each directory
dataframes = []
for dire in args.directories:
    label = dir_label_map[dire]
    dir_path = os.path.join("results", dire)
    # Look for files ending with '_results_Presence.csv'
    result_files = [f for f in os.listdir(dir_path) if f.endswith('_results_Presence.csv')]
    if not result_files:
        raise FileNotFoundError(f"No presence result files found in directory: {dir_path}")
    for result_file in result_files:
        file_path = os.path.join(dir_path, result_file)
        df_temp = pd.read_csv(file_path)
        # Extract model name from filename (assumes a name like 'gpt-4o_results_Presence.csv')
        model_name = result_file.replace('_results_Presence.csv', '')
        df_temp['label'] = label
        df_temp['model'] = model_name
        dataframes.append(df_temp)

# Combine all dataframes into one
df = pd.concat(dataframes, ignore_index=True)

# Ensure required columns are present
required_columns = ['selected_presence', 'actual_presence', 'num_distractors', 'colourbin']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"The following required columns are missing from the DataFrame: {missing_columns}")

# Convert presence columns to numeric values (coercing errors to NaN)
df['selected_presence_numeric'] = pd.to_numeric(df['selected_presence'], errors='coerce')
df['actual_presence_numeric'] = pd.to_numeric(df['actual_presence'], errors='coerce')

# Create a 'correct' column: True if selected equals actual, False otherwise.
df['correct'] = df['selected_presence_numeric'] == df['actual_presence_numeric']

# Prepare an output text string to save analysis results.
output_text = ""
label_model_groups = df.groupby(['label', 'model'])
for (label, model), df_group in label_model_groups:
    total_predictions = len(df_group)
    correct_predictions = df_group['correct'].sum()
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    report_text = (
        f"Label: {label}, Model: {model}\n"
        f"Total Predictions: {total_predictions}\n"
        f"Correct Predictions: {correct_predictions}\n"
        f"Accuracy: {accuracy * 100:.2f}%\n\n"
    )
    print(report_text)
    output_text += report_text

    # Drop rows with NaN in presence columns before computing classification report.
    valid_idx = df_group.dropna(subset=['selected_presence_numeric', 'actual_presence_numeric']).index
    if not valid_idx.empty:
        class_report = classification_report(
            df_group.loc[valid_idx, 'actual_presence_numeric'],
            df_group.loc[valid_idx, 'selected_presence_numeric'],
            zero_division=0
        )
        report = f"Classification Report for Label: {label}, Model: {model}:\n{class_report}\n\n"
        print(report)
        output_text += report

# Save the text output to a file
if save_figures:
    text_output_file = os.path.join(output_dir, 'presence_analysis_results.txt')
else:
    text_output_file = os.path.join(os.getcwd(), 'presence_analysis_results.txt')
with open(text_output_file, 'w') as f:
    f.write(output_text)

# Compute accuracy versus number of distractors for each label and model
accuracy_vs_distractor_data = []
for (label, model), df_group in label_model_groups:
    unique_ks = sorted(df_group['num_distractors'].unique())
    for k in unique_ks:
        df_k = df_group[df_group['num_distractors'] == k]
        total_k = len(df_k)
        correct_k = df_k['correct'].sum()
        accuracy_k = correct_k / total_k if total_k > 0 else 0
        # Compute standard error (if more than one sample is available)
        std_error_k = np.sqrt((accuracy_k * (1 - accuracy_k)) / total_k) if total_k > 1 else 0
        accuracy_vs_distractor_data.append({
            'Label': label,
            'Model': model,
            'Number of Distractors (k)': k,
            'Accuracy': accuracy_k,
            'Standard Error': std_error_k
        })

accuracy_vs_distractor_df = pd.DataFrame(accuracy_vs_distractor_data)
accuracy_vs_distractor_df['Label_Model'] = accuracy_vs_distractor_df['Label'] + ' - ' + accuracy_vs_distractor_df['Model']

# Plot Accuracy vs. Number of Distractors with error bounds
plt.figure(figsize=(10, 6))
max_width = 15
for label_model, group in accuracy_vs_distractor_df.groupby('Label_Model'):
    k_values = group['Number of Distractors (k)']
    accuracy_values = group['Accuracy']
    std_errors = group['Standard Error']
    # Compute 95% confidence bounds (ensuring they stay in the [0,1] interval)
    upper_bound = np.minimum(accuracy_values + 1.96 * std_errors, 1)
    lower_bound = np.maximum(accuracy_values - 1.96 * std_errors, 0)
    wrapped_label = "\n".join(textwrap.wrap(label_model, width=max_width))
    plt.plot(k_values, accuracy_values, '-o', label=wrapped_label)
    plt.fill_between(k_values, lower_bound, upper_bound, alpha=0.2)

plt.xlabel('Number of Distractors (k)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Number of Distractors (Presence Mode)')
plt.ylim(0, 1)
plt.grid(True)
plt.legend(title='Label - Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
if save_figures:
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_distractors_presence.png'), bbox_inches='tight')
plt.show()
