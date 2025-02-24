import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directories", nargs="+", required=True,
                    help="List of directories containing results CSV files.")
parser.add_argument("-l", "--labels", nargs="+", required=True,
                    help="List of labels corresponding to each directory.")
args = parser.parse_args()

# Ensure the number of directories matches the number of labels
if len(args.directories) != len(args.labels):
    raise ValueError("The number of directories and labels must be the same.")

# Map each directory to its corresponding label
dir_label_map = dict(zip(args.directories, args.labels))

# Initialize a list to hold all grouped DataFrames
all_grouped = []

# Process each directory
for dire in args.directories:
    dir_path = "results/" + dire
    label = dir_label_map[dire]
    
    # Get list of result files in the directory
    result_files = [f for f in os.listdir(dir_path) if f.endswith('_results_Coords.csv')]
    if not result_files:
        raise FileNotFoundError(f"No result files found in directory: {dir_path}")
    
    for result_file in result_files:
        file_path = os.path.join(dir_path, result_file)
        
        # Load the results into a pandas DataFrame
        df = pd.read_csv(file_path)
        
        # Extract the model name from the filename
        # Assuming filenames are like 'gpt-4o_results.csv', 'claude-sonnet_results.csv'
        model_name = result_file.replace('_results.csv', '')
        
        # Remove rows with missing Euclidean error values and create a copy to avoid warnings
        df_clean = df.dropna(subset=['euclidean_error']).copy()
        
        # Ensure 'num_distractors' is a numeric type
        df_clean['num_distractors'] = pd.to_numeric(df_clean['num_distractors'], errors='coerce')
        
        # Remove any rows where 'num_distractors' could not be converted to a number
        df_clean = df_clean.dropna(subset=['num_distractors'])
        
        # Convert 'num_distractors' to integer type
        df_clean['num_distractors'] = df_clean['num_distractors'].astype(int)
        
        # Add columns to indicate the label and model
        df_clean['label'] = label
        df_clean['model'] = model_name
        
        # Group by 'num_distractors' and compute the average Euclidean error
        grouped = df_clean.groupby(['label', 'model', 'num_distractors'], as_index=False)['euclidean_error'].mean()
        
        # Append the grouped DataFrame to the list
        all_grouped.append(grouped)

# Concatenate all grouped DataFrames
combined_grouped = pd.concat(all_grouped, ignore_index=True)

# Combine Label and Model for plotting
combined_grouped['Label_Model'] = combined_grouped['label'] + ' - ' + combined_grouped['model']

# Set the style for the plot
sns.set(style="whitegrid")

# Create a figure and axis
plt.figure(figsize=(12, 6))

# Plot using seaborn with 'Label_Model' as the hue
sns.lineplot(
    data=combined_grouped,
    x='num_distractors',
    y='euclidean_error',
    hue='Label_Model',
    marker='o',
    linewidth=2,
    sort=False
)

# Add labels and title
plt.xlabel('Number of Distractors', fontsize=14)
plt.ylabel('Average Euclidean Error', fontsize=14)
plt.title('Average Euclidean Error vs. Number of Distractors', fontsize=16)

# Determine the full range of x-axis values
min_distractors = combined_grouped['num_distractors'].min()
max_distractors = combined_grouped['num_distractors'].max()

# Set a maximum number of ticks (for example, 10) and calculate an appropriate step size
max_ticks = 10
tick_range = max_distractors - min_distractors
step = max(1, tick_range // max_ticks)

# Set x-axis ticks with the calculated step size
plt.xticks(range(min_distractors, max_distractors + 1, step))

# Adjust legend
plt.legend(title='Label - Model', fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()
