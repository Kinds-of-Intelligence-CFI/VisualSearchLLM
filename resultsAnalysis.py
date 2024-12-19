import os
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse

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

# Read in data from multiple directories and models
dataframes = []
for dire in args.directories:
    label = dir_label_map[dire]
    dir_path = "results/"+dire
    # Get list of result files in the directory
    result_files = [f for f in os.listdir(dir_path) if f.endswith('_results_Quadrant.csv')]
    if not result_files:
        raise FileNotFoundError(f"No result files found in directory: {dir_path}")
    for result_file in result_files:
        file_path = os.path.join(dir_path, result_file)

        # Load the results into a pandas DataFrame
        df_temp = pd.read_csv(file_path)

        # Extract the model name from the filename
        # Assuming filenames are like 'gpt-4o_results.csv', 'claude-sonnet_results.csv'
        model_name = result_file.replace('_results.csv', '')

        # Add columns to indicate the label and model
        df_temp['label'] = label
        df_temp['model'] = model_name

        dataframes.append(df_temp)

# Combine all dataframes into one
df = pd.concat(dataframes, ignore_index=True)

# Convert 'correct' column to boolean if it's not already
df['correct'] = df['correct'].astype(bool)

# Ensure necessary columns are in the DataFrame
required_columns = ['num_distractors', 'colourbin', 'actual_quadrant', 'selected_quadrant']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"The following required columns are missing from the DataFrame: {missing_columns}")

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
        accuracy = accuracy_score(df_group['actual_quadrant'], df_group['selected_quadrant'])

        result_text = (f"Label: {label}, Model: {model}\n"
                       f"Total predictions: {total_predictions}\n"
                       f"Correct predictions: {correct_predictions}\n"
                       f"Overall Accuracy: {accuracy * 100:.2f}%\n\n")
        print(result_text)
        file.write(result_text)

        # Overall classification report
        classification_report_text = classification_report(
            df_group['actual_quadrant'], df_group['selected_quadrant'], zero_division=0
        )
        print(f"Overall Classification Report for Label: {label}, Model: {model}:\n", classification_report_text)
        file.write(f"Overall Classification Report for Label: {label}, Model: {model}:\n" + classification_report_text + "\n\n")

# Calculate success rate per quadrant per label per model
success_rates_list = []
grouped = df.groupby(['label', 'model'])

for (label, model), df_group in grouped:
    quadrants = ['Quadrant 1', 'Quadrant 2', 'Quadrant 3', 'Quadrant 4']
    success_rate_text = f"Average Success Rate per Quadrant for Label: {label}, Model: {model}:\n"
    for quadrant in quadrants:
        total_actual = sum(df_group['actual_quadrant'] == quadrant)
        correct_predictions = sum(
            (df_group['actual_quadrant'] == quadrant) & (df_group['selected_quadrant'] == quadrant)
        )
        success_rate = correct_predictions / total_actual if total_actual > 0 else 0
        success_rates_list.append({
            'Label': label,
            'Model': model,
            'Quadrant': quadrant,
            'Success Rate': success_rate
        })
        success_rate_text += f"{quadrant}: {success_rate * 100:.2f}%\n"
    print(success_rate_text)
    with open(text_output_file, 'a') as file:
        file.write(success_rate_text)

# Create DataFrame from success_rates_list
success_rates_df = pd.DataFrame(success_rates_list)

# Combine Label and Model for plotting
success_rates_df['Label_Model'] = success_rates_df['Label'] + ' - ' + success_rates_df['Model']

# Plot success rate per quadrant per label per model
g = sns.catplot(
    x='Quadrant', y='Success Rate', hue='Label_Model',
    data=success_rates_df, kind='bar', palette='viridis', height=6, aspect=1.5
)
g.set(ylim=(0, 1))
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Success Rate per Quadrant per Label and Model', fontsize=16)
plt.legend(title='Label - Model', bbox_to_anchor=(1.05, 1), loc='upper left')

if save_figures:
    plt.savefig(os.path.join(output_dir, 'success_rate_per_quadrant_per_label_model.png'), bbox_inches='tight')
plt.show()

# Initialize list to hold accuracy vs distractor data
accuracy_vs_distractor_data = []

# Calculate accuracy and standard error for each number of distractors (k) per label per model
for (label, model), df_group in grouped:
    unique_ks = sorted(df_group['num_distractors'].unique())
    for k in unique_ks:
        df_k = df_group[df_group['num_distractors'] == k]
        total_k = len(df_k)
        correct_k = df_k['correct'].sum()
        accuracy_k = accuracy_score(df_k['actual_quadrant'], df_k['selected_quadrant'])

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

# Plot accuracy vs number of distractors without error bars per label per model
g = sns.catplot(
    x='Number of Distractors (k)', y='Accuracy', hue='Label_Model',
    data=accuracy_vs_distractor_df, kind='bar', palette='viridis', height=6, aspect=1.5
)
g.set(ylim=(0, 1))
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Accuracy vs. Number of Distractors (k) per Label and Model', fontsize=16)
plt.legend(title='Label - Model', bbox_to_anchor=(1.05, 1), loc='upper left')

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
    upper_bound = np.minimum(accuracy_values + std_errors, 1)
    lower_bound = np.maximum(accuracy_values - std_errors, 0)

    # Plot the accuracy line
    plt.plot(k_values, accuracy_values, '-o', label=label_model)

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
    # Define the quadrant labels
    quadrant_labels = ['Quadrant 1', 'Quadrant 2', 'Quadrant 3', 'Quadrant 4']

    for (label, model), df_group in grouped:
        unique_ks = sorted(df_group['num_distractors'].unique())
        for k in unique_ks:
            df_k = df_group[df_group['num_distractors'] == k]

            if len(df_k) == 0:
                continue  # Skip if there is no data for this number of distractors

            # Compute the confusion matrix
            cm = confusion_matrix(
                df_k['actual_quadrant'],
                df_k['selected_quadrant'],
                labels=quadrant_labels
            )

            # Create a DataFrame for the confusion matrix
            cm_df = pd.DataFrame(cm, index=quadrant_labels, columns=quadrant_labels)

            # Plot the confusion matrix as a heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix (Label: {label}, Model: {model}, k = {k})')
            plt.xlabel('Predicted Quadrant')
            plt.ylabel('Actual Quadrant')

            # Save the confusion matrix plot
            if save_figures:
                cm_filename = os.path.join(
                    output_dir, f'confusion_matrix_{label}_{model}_k_{k}.png'
                )
                plt.savefig(cm_filename)
            plt.close()  # Close the figure to free up memory

            # Optionally, print the confusion matrix to the terminal and save to text file
            cm_text = f"\nConfusion Matrix for Label: {label}, Model: {model}, k = {k}:\n{cm_df}\n"
            print(cm_text)
            with open(text_output_file, 'a') as file:
                file.write(cm_text)
