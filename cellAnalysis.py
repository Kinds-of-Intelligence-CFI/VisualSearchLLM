import os
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
import re
import math
import matplotlib.gridspec as gridspec
import textwrap
# Argument parsing to accept multiple directories and labels
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directories", nargs="+", required=True,
                    help="List of directories containing results CSV files")
parser.add_argument("-l", "--labels", nargs="+", required=True,
                    help="List of labels for each directory.")
parser.add_argument("-c", "--confusion", action='store_true',
                    help="Include to compute and save confusion matrices.")
args = parser.parse_args()

# Ensure the number of directories matches the number of labels
if len(args.directories) != len(args.labels):
    raise ValueError("The number of directories and labels must be the same.")

# Determine if we should save figures and where
if len(args.directories) == 1:
    save_figures = True
    output_dir = "results/"+args.directories[0]
else:
    save_figures = False
    output_dir = None

# Map each directory to its corresponding label
dir_label_map = dict(zip(args.directories, args.labels))


max_width = 15

# Read in data from multiple directories
dataframes = []
for dire in args.directories:
    label = dir_label_map[dire]
    dir_path = "results/"+dire
    # Get list of result files in the directory

    result_files = [f for f in os.listdir(dir_path) if f.endswith('_results_Cells.csv')]
    if not result_files:
        raise FileNotFoundError(f"No result files found in directory: {dir_path}")
    for result_file in result_files:
        file_path = os.path.join(dir_path, result_file)

        # Load the results into a pandas DataFrame
        df_temp = pd.read_csv(file_path)

        # Extract the model name from the filename
        # Assuming filenames are like 'gpt4o_results.csv', 'claude_results.csv'
        model_name = result_file.replace('_results_Cell.csv', '')

        # Check for required columns
        if 'selected_cell' not in df_temp.columns:
            raise ValueError(f"Column 'selected_cell' not found in file '{result_file}'")
        if 'correct' not in df_temp.columns:
            raise ValueError(f"Column 'correct' not found in file '{result_file}'")

        # Rename 'selected_cell' to 'predicted_cell'
        df_temp.rename(columns={'selected_cell': 'predicted_cell'}, inplace=True)

        # Add columns to indicate the label and model
        df_temp['label'] = label
        df_temp['model'] = model_name

        dataframes.append(df_temp)

# Combine all dataframes into one
df = pd.concat(dataframes, ignore_index=True)

# Convert 'correct' column to boolean if it's not already
df['correct'] = df['correct'].astype(bool)

# Ensure necessary columns are in the DataFrame
required_columns = ['num_distractors', 'colourbin', 'actual_cell', 'predicted_cell']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"The following required columns are missing from the DataFrame: {missing_columns}")

# Function to parse cell string into (row, column) tuple
def parse_cell(cell_str):
    match = re.match(r'\((\d+),\s*(\d+)\)', str(cell_str))
    if match:
        row = int(match.group(1))
        col = int(match.group(2))
        return (row, col)
    else:
        return None

# Parse 'actual_cell' and 'predicted_cell'
df['actual_cell_parsed'] = df['actual_cell'].apply(parse_cell)
df['predicted_cell_parsed'] = df['predicted_cell'].apply(parse_cell)

# Handle rows where parsing failed
df = df.dropna(subset=['actual_cell_parsed', 'predicted_cell_parsed'])

# Create a set of unique cells based only on 'actual_cell_parsed'
unique_cells = set(df['actual_cell_parsed'])

# Create sorted list of cell labels
cell_labels = sorted(unique_cells)
cell_labels_str = [f'Cell ({cell[0]}, {cell[1]})' for cell in cell_labels]
cell_to_label_str = {cell: f'Cell ({cell[0]}, {cell[1]})' for cell in cell_labels}

# Map 'actual_cell_parsed' to labels
df['actual_cell_label'] = df['actual_cell_parsed'].map(cell_to_label_str)

# Map 'predicted_cell_parsed' to labels, handling invalid predictions
def map_predicted_cell(cell):
    if cell in cell_to_label_str:
        return cell_to_label_str[cell]
    else:
        return 'Invalid Prediction'

df['predicted_cell_label'] = df['predicted_cell_parsed'].apply(map_predicted_cell)

# Create text output file
if save_figures:
    text_output_file = os.path.join(output_dir, 'analysis_results.txt')
else:
    text_output_file = os.path.join(os.getcwd(), 'analysis_results.txt')

with open(text_output_file, 'w') as file:
    # Calculate overall accuracy per label per model
    label_model_groups = df.groupby(['label', 'model'])
    for (label, model), df_group in label_model_groups:
        total_predictions = len(df_group)
        correct_predictions = df_group['correct'].sum()
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        result_text = (f"Label: {label}, Model: {model}\n"
                       f"Total predictions: {total_predictions}\n"
                       f"Correct predictions: {correct_predictions}\n"
                       f"Overall Accuracy: {accuracy * 100:.2f}%\n\n")
        print(result_text)
        file.write(result_text)

        # Overall classification report
        classification_report_text = classification_report(
            df_group['actual_cell_label'], df_group['predicted_cell_label'], zero_division=0
        )
        print(f"Overall Classification Report for Label: {label}, Model: {model}:\n", classification_report_text)
        file.write(f"Overall Classification Report for Label: {label}, Model: {model}:\n" + classification_report_text + "\n\n")

# Calculate success rate per cell per label per model
success_rates_list = []
for (label, model), df_group in label_model_groups:
    success_rate_text = f"Average Success Rate per Cell for Label: {label}, Model: {model}:\n"
    for cell_label, cell in zip(cell_labels_str, cell_labels):
        total_actual = sum(df_group['actual_cell_label'] == cell_label)
        correct_predictions = sum(
            (df_group['actual_cell_label'] == cell_label) & (df_group['predicted_cell_label'] == cell_label)
        )
        success_rate = correct_predictions / total_actual if total_actual > 0 else 0
        success_rates_list.append({
            'Label': label,
            'Model': model,
            'Cell': cell_label,
            'Success Rate': success_rate
        })
        success_rate_text += f"{cell_label}: {success_rate * 100:.2f}%\n"
    print(success_rate_text)
    with open(text_output_file, 'a') as file:
        file.write(success_rate_text)

# Create DataFrame from success_rates_list
success_rates_df = pd.DataFrame(success_rates_list)

# Combine Label and Model for plotting
success_rates_df['Label_Model'] = success_rates_df['Label'] + ' - ' + success_rates_df['Model']

# Plot success rate per cell as heatmaps for each Label_Model
label_model_pairs = success_rates_df['Label_Model'].unique()
num_plots = len(label_model_pairs)

# Determine the grid size for subplots
cols = math.ceil(math.sqrt(num_plots))
rows = math.ceil(num_plots / cols)

fig = plt.figure(figsize=(5 * cols, 5 * rows))
gs = gridspec.GridSpec(rows, cols)

for idx, label_model in enumerate(label_model_pairs):
    ax = fig.add_subplot(gs[idx])
    df_pair = success_rates_df[success_rates_df['Label_Model'] == label_model].copy()

    # Parse 'Cell' into 'row' and 'column'
    df_pair['row'] = df_pair['Cell'].apply(lambda x: int(re.search(r'\((\d+),', x).group(1)))
    df_pair['column'] = df_pair['Cell'].apply(lambda x: int(re.search(r',\s*(\d+)\)', x).group(1)))

    max_row = df_pair['row'].max()
    max_col = df_pair['column'].max()

    # Create a pivot table with rows and columns
    pivot_table = df_pair.pivot(index='row', columns='column', values='Success Rate')

    # Reindex to ensure all rows and columns are present
    pivot_table = pivot_table.reindex(index=range(1, max_row + 1), columns=range(1, max_col + 1))

    # Plot the heatmap
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap='viridis', ax=ax, vmin=0, vmax=1)
    ax.set_title(f'Success Rate for {label_model}')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

plt.tight_layout()

if save_figures:
    plt.savefig(os.path.join(output_dir, 'success_rate_per_cell_heatmap.png'))

plt.show()

# Initialize list to hold accuracy vs distractor data
accuracy_vs_distractor_data = []

# Calculate accuracy and standard error for each number of distractors (k) per label per model
for (label, model), df_group in label_model_groups:
    unique_ks = sorted(df_group['num_distractors'].unique())
    for k in unique_ks:
        df_k = df_group[df_group['num_distractors'] == k]
        total_k = len(df_k)
        correct_k = df_k['correct'].sum()
        accuracy_k = correct_k / total_k if total_k > 0 else 0

        # Calculate standard error
        if total_k > 1:
            std_error_k = np.sqrt((accuracy_k * (1 - accuracy_k)) / total_k)
        else:
            std_error_k = 0  # No error if only one sample

        accuracy_vs_distractor_data.append({
            'Label': label,
            'Model': model,
            'Number of Distractors (k)': k,
            'Accuracy': accuracy_k,
            'Standard Error': std_error_k
        })

        # Save metrics for each k in the text file
        k_text = (f"\n===== Metrics for Label: {label}, Model: {model}, k = {k} ({total_k} samples) =====\n"
                  f"Correct predictions: {correct_k}\n"
                  f"Accuracy: {accuracy_k * 100:.2f}%\n"
                  f"Standard Error: {std_error_k:.4f}\n")
        print(k_text)
        with open(text_output_file, 'a') as file:
            file.write(k_text)

# Convert data into DataFrame
accuracy_vs_distractor_df = pd.DataFrame(accuracy_vs_distractor_data)

# Combine Label and Model for plotting
accuracy_vs_distractor_df['Label_Model'] = accuracy_vs_distractor_df['Label'] + ' - ' + accuracy_vs_distractor_df['Model']


if save_figures:
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_number_of_distractors.png'), bbox_inches='tight')
plt.show()

# Plot line plot for accuracy vs number of distractors with shaded error region per label per model
plt.figure(figsize=(10, 6))
for label_model, df_group in accuracy_vs_distractor_df.groupby('Label_Model'):
    k_values = df_group['Number of Distractors (k)']
    accuracy_values = df_group['Accuracy']
    std_errors = df_group['Standard Error']

    # Corrected: Ensure the lower and upper bounds are within [0, 1]
    upper_bound = np.minimum(accuracy_values + 1.96*std_errors, 1)
    lower_bound = np.maximum(accuracy_values - 1.96*std_errors, 0)

    # Plot the accuracy line
    wrapped_label = "\n".join(textwrap.wrap(label_model, width=max_width))
    plt.plot(k_values, accuracy_values, '-o', label=wrapped_label)

    # Fill between the upper and lower bounds
    plt.fill_between(k_values, lower_bound, upper_bound, alpha=0.2)

plt.xlabel('Number of Distractors (k)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Number of Distractors (k) with Shaded Error Region per Label and Model')
plt.ylim(0, 1)
plt.grid(True)
plt.legend(title='Label - Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

if save_figures:
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_number_of_distractors_shaded_error.png'), bbox_inches='tight')
plt.show()

# Compute and save confusion matrices only if the -c or --confusion flag is set
if args.confusion:
    # Include 'Invalid Prediction' in the labels
    cell_labels_order = cell_labels_str + ['Invalid Prediction']

    for (label, model), df_group in label_model_groups:
        unique_ks = sorted(df_group['num_distractors'].unique())
        for k in unique_ks:
            df_k = df_group[df_group['num_distractors'] == k]

            if len(df_k) == 0:
                continue  # Skip if there is no data for this number of distractors

            # Compute the confusion matrix
            cm = confusion_matrix(
                df_k['actual_cell_label'],
                df_k['predicted_cell_label'],
                labels=cell_labels_order
            )

            # Create a DataFrame for the confusion matrix
            cm_df = pd.DataFrame(cm, index=cell_labels_order, columns=cell_labels_order)

            # Plot the confusion matrix as a heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix for Cells (Label: {label}, Model: {model}, k = {k})')
            plt.xlabel('Predicted Cell')
            plt.ylabel('Actual Cell')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)

            plt.tight_layout()

            # Save the confusion matrix plot
            if save_figures:
                cm_filename = os.path.join(
                    output_dir, f'confusion_matrix_cells_{label}_{model}_k_{k}.png'
                )
                plt.savefig(cm_filename)
            plt.close()  # Close the figure to free up memory

            # Optionally, print the confusion matrix to the terminal and save to text file
            cm_text = f"\nConfusion Matrix for Label: {label}, Model: {model}, k = {k}:\n{cm_df}\n"
            print(cm_text)
            with open(text_output_file, 'a') as file:
                file.write(cm_text)
