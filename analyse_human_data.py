import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

path = os.getcwd()
results_path = os.path.join(path, r"human_results")

def clean_columns(df):
    df = df[(df["Display"] == "Task") & (df["Screen"] == "trial")]
    spreadsheet_renames = {
        col: col.split("Spreadsheet: ")[1]
        for col in df.columns
        if col.startswith("Spreadsheet: ")
    }
    fixed_renames = {
        "Participant Public ID": "PID",
        "Trial Number": "trial",
        "Response": "response",
        "Reaction Time": "rt",
        "Correct": "accuracy"
    }
    all_renames = {**fixed_renames, **spreadsheet_renames}
    df = df[list(all_renames.keys())]
    df.rename(columns=all_renames, inplace=True)
    df.rename({"answer":"correct_answer"}, inplace=True)
    df['accuracy'] = df['accuracy'].fillna(0).astype(int)
    return df

## load results
experiments = ["e1_numbers",
               "e2_light_priors",
               "e3_circle_sizes"
]
participant_list = []
for selected_experiment in range(len(experiments)):
    experiment = experiments[selected_experiment]
    df = pd.read_csv(os.path.join(results_path, f"{experiment}.csv"))
    df = clean_columns(df)
    if selected_experiment == 0:
        condition = "colour_type"
        df['colour_type'] = df['colour_type'].replace({
            'no_colour': 'Inefficient disjunctive',
            'colour': 'Efficient disjunctive',
            'conjunctive': 'Conjunctive'
        })
        bin_edges = [1, 5, 9, 17, 33, 65, 100]
        bin_labels = [
            '1–4',
            '5–8',
            '9–16',
            '17–32',
            '33–64',
            '65–99'
        ]

    elif selected_experiment == 1:
        condition = "light_direction"
        bin_edges = [1, 5, 9, 13, 17, 21, 25, 33, 50]
        bin_labels = [
            '1–4',
            '5–8',
            '9–12',
            '13–16',
            '17–20',
            '21–24',
            '25-32',
            '33-49'
        ]
    elif selected_experiment == 2:
        condition = "target_size"
        bin_edges = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 50]
        bin_labels = [
            '1–4',
            '5–8',
            '9–12',
            '13–16',
            '17–20',
            '21–24',
            '25-28',
            '29-32',
            '33-36',
            '37-40',
            '41-44',
            '45-49'
        ]
    df['distractor_bin'] = pd.cut(df['num_distractors'],
                                  bins=bin_edges,
                                  labels=bin_labels,
                                  right=False)

    ## remove low scoring participants
    def find_valid_participants(df, accuracy_threshold=0.25):
        df = df.copy()
        participant_accuracy = df.groupby('PID')['accuracy'].mean()
        valid_participants = participant_accuracy[participant_accuracy >= accuracy_threshold].index
        total_participants = len(participant_accuracy)
        filtered_participants = total_participants - len(valid_participants)
        print(f"Total participants: {total_participants}")
        print(f"Participants below threshold ({accuracy_threshold*100}%): {filtered_participants}")
        print(f"Participants remaining: {len(valid_participants)}")
        return list(valid_participants)

    accuracy_threshold = 0.25
    valid_participants = find_valid_participants(df,accuracy_threshold)
    participant_list += valid_participants
    df_filtered = df[df['PID'].isin(valid_participants)]

    ## save
    df_filtered.to_csv(os.path.join(results_path, f"{experiment}_processed.csv"), index=False)

    ## plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_filtered,
        x='distractor_bin',
        y='accuracy',
        hue=condition,
        errorbar=('ci', 95),
        marker='o'
    )
    plt.title(f"Participants with >{accuracy_threshold*100}% Accuracy (n={len(valid_participants)})")
    plt.xlabel("Number of Distractors")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.legend(title='Label')
    plt.show()

