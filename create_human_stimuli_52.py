import os
import pandas as pd
import numpy as np
from PIL import Image
import random
from collections import Counter

path = os.getcwd()
stim_path = os.path.join(path, "original_stimuli")
new_stim_path = os.path.join(path, "human_stimuli")

## experiment parameters
def sample_from_bins(bins, seed=None, samples_per_bin=4):
    rng = random.Random(seed)
    all_samples = []
    for i, bin_range in enumerate(bins):
        bin_values = list(bin_range)
        if len(bin_values) < samples_per_bin:
            raise ValueError(f"Bin {i + 1} has only {len(bin_values)} values, but {samples_per_bin} samples requested.")
        samples = rng.sample(bin_values, k=samples_per_bin)
        all_samples.extend(samples)
    return all_samples

random_seed = 42
distractor_bins = [
    range(1, 5),  # 1 to 4
    range(5, 9),  # 5 to 8
    range(9, 17),  # 9 to 16
    range(17, 33),  # 17 to 32
    range(33, 65),  # 33 to 64
    range(65, 100)  # 65 to 99
]
samples_per_bin = 4
distractor_nums = sample_from_bins(
    distractor_bins,
    seed=random_seed,
    samples_per_bin=samples_per_bin
)
types_of_stimuli = [
    "2Among5Colour",
    "2Among5NoColour",
    "2Among5Conj",
    "5Among2Colour",
    "5Among2NoColour",
    "5Among2Conj"
]
quadrants = [
    "Quadrant 1",
    "Quadrant 2",
    "Quadrant 3",
    "Quadrant 4"
]
colours = [
    "#FF0000",
    "#00FF00",
    "#0000FF"
]

def get_bin_index(num, bins):
    for i, bin_range in enumerate(bins):
        if num in bin_range:
            return i
    return None

def quadrant_balanced_sample(data, distractor_nums, distractor_bins, balance=None):
    data = data.copy()
    result = pd.DataFrame()
    distractor_to_bin = {num: get_bin_index(num, distractor_bins) for num in distractor_nums}
    bins_to_distractor_nums = {}
    for num in distractor_nums:
        bin_idx = distractor_to_bin[num]
        if bin_idx not in bins_to_distractor_nums:
            bins_to_distractor_nums[bin_idx] = []
        bins_to_distractor_nums[bin_idx].append(num)
    rng = random.Random(random_seed)
    color_counter = Counter()
    for bin_idx, nums_in_bin in bins_to_distractor_nums.items():
        bin_data = data[data["num_distractors"].isin(nums_in_bin)]
        if len(bin_data) < 4:
            print(
                f"WARNING: Not enough samples for bin {bin_idx} (range: {list(distractor_bins[bin_idx])[0]}-{list(distractor_bins[bin_idx])[-1]})")
            continue
        random_quadrant_order = quadrants.copy()
        rng.shuffle(random_quadrant_order)
        for i, num in enumerate(nums_in_bin):
            quadrant = random_quadrant_order[i % len(quadrants)]
            num_quadrant_data = bin_data[(bin_data["num_distractors"] == num) & (bin_data["quadrant"] == quadrant)]
            if num_quadrant_data.empty:
                available_quadrants = bin_data[bin_data["num_distractors"] == num]["quadrant"].unique()
                if len(available_quadrants) == 0:
                    print(f"WARNING: No samples for distractor number {num}")
                    continue
                quadrant = rng.choice(available_quadrants)
                num_quadrant_data = bin_data[(bin_data["num_distractors"] == num) & (bin_data["quadrant"] == quadrant)]
            num_quadrant_data = num_quadrant_data.copy()
            if balance:
                num_quadrant_data['balance_score'] = num_quadrant_data.apply(
                    lambda row: -color_counter[row[balance]],
                    axis=1)
                num_quadrant_data = num_quadrant_data.sample(frac=1, random_state=rng.randint(0, 10000))
                num_quadrant_data = num_quadrant_data.sort_values('balance_score', ascending=False)
            else:
                num_quadrant_data = num_quadrant_data.sample(frac=1, random_state=rng.randint(0, 10000))
            if not num_quadrant_data.empty:
                selected_row = num_quadrant_data.iloc[0]
                if balance:
                    color_counter[selected_row[balance]] += 1
                if 'balance_score' in selected_row:
                    selected_row = selected_row.drop('balance_score')
                result = pd.concat([result, pd.DataFrame([selected_row])], ignore_index=True)
    if balance:
        distribution = result[balance].value_counts()
        print(f"{balance} distribution in selected samples: {dict(distribution)}")
    for bin_idx in bins_to_distractor_nums.keys():
        bin_nums = bins_to_distractor_nums[bin_idx]
        bin_result = result[result["num_distractors"].isin(bin_nums)]
        quadrant_dist = bin_result["quadrant"].value_counts().to_dict()
        print(
            f"Bin {bin_idx} ({list(distractor_bins[bin_idx])[0]}-{list(distractor_bins[bin_idx])[-1]}) quadrant distribution: {quadrant_dist}")
    return result

def select_different_distractor(row):
    for color in row['distractor_colors']:
        if color != row['target_color']:
            return color
    return None

# sample from stimuli
conjunctive_types = ["2Among5Conj", "5Among2Conj"]
cols = ['filename','shape_type','center_x','center_y','rotation_angle','color','quadrant','num_distractors']
new_df = pd.DataFrame()
for st in types_of_stimuli:
    directory = os.path.join(stim_path, st)
    df = pd.read_csv(os.path.join(directory, "annotations.csv"))
    df["color"] = df["color"].str.upper()
    df_targets = df[df['target'] == True][cols].copy()
    df_targets.rename(columns={'color': 'target_color'}, inplace=True)
    df_targets = df_targets[~((df_targets['center_x'].between(170, 230)) | (df_targets['center_y'].between(170, 230)))]
    df_distractors = df[df['target'] == False].groupby('filename')['color'].unique().reset_index()
    df_distractors.rename(columns={'color': 'distractor_colors'}, inplace=True)
    df_merged = pd.merge(df_targets, df_distractors, on='filename', how='left')
    if st in conjunctive_types:
        df_merged = df_merged[df_merged.apply(
            lambda row: (
                    isinstance(row['distractor_colors'], np.ndarray) and
                    (len(np.unique(row['distractor_colors'])) >= 2 or row['num_distractors'] == 1)
            ),
            axis=1
        )]
        df_merged['distractor_color'] = df_merged.apply(
            lambda row: (
                select_different_distractor(row)
                if row['num_distractors'] > 1 else row['distractor_colors'][0]
            ),
            axis=1
        )
        df_merged = df_merged[df_merged['target_color'] != df_merged['distractor_color']]
    else:
        df_merged = df_merged[df_merged['distractor_colors'].apply(
            lambda x: isinstance(x, np.ndarray) and len(np.unique(x)) == 1
        )]
        df_merged['distractor_color'] = df_merged['distractor_colors'].apply(lambda x: x[0] if len(x) > 0 else None)
    df_merged["type"] = st
    df_merged = df_merged[df_merged["num_distractors"].isin(distractor_nums)]
    df_merged["colors_combined"] = df_merged['target_color'] + df_merged['distractor_color'].fillna("")
    temp_df = quadrant_balanced_sample(
        data=df_merged,
        distractor_nums=distractor_nums,
        distractor_bins=distractor_bins,
        balance="colors_combined"
    )
    new_df = pd.concat([new_df, temp_df], ignore_index=True)

## load and save images
for i, file in enumerate(new_df["filename"]):
    directory = os.path.join(stim_path, new_df.loc[i, "type"])
    img = Image.open(os.path.join(directory, file))
    img_name = f"image_{i}.jpg"
    new_df.loc[i, "filename_new"] = img_name
    img.convert("RGB").save(os.path.join(new_stim_path, img_name))
new_df.to_csv(os.path.join(new_stim_path, "human_stimuli.csv"), index=False)

## format df for gorilla
condition_dict = {
    "2Among5Colour":"colour",
    "2Among5NoColour":"no_colour",
    "2Among5Conj":"conjunctive",
    "5Among2Colour":"colour",
    "5Among2NoColour":"no_colour",
    "5Among2Conj":"conjunctive"
}
colour_dict = {
    "#FF0000":"red",
    "#00FF00":"green",
    "#0000FF":"blue"
}
gorilla_cols = ['filename','shape_type','colour_type','target_color',
                'distractor_color','quadrant','num_distractors','answer']
gorilla_df = new_df.copy()
gorilla_df.drop(labels=["filename"], axis=1, inplace=True)
gorilla_df.rename(columns={'filename_new': 'filename'}, inplace=True)
gorilla_df['answer'] = gorilla_df['quadrant'].str[-1].astype(int)
gorilla_df['colour_type'] = gorilla_df['type'].replace(condition_dict)
gorilla_df['target_color'] = gorilla_df['target_color'].replace(colour_dict)
gorilla_df['distractor_color'] = gorilla_df['distractor_color'].replace(colour_dict)
gorilla_df = gorilla_df[gorilla_cols]
intro_row = pd.DataFrame([['Introduction'] + [''] * len(gorilla_df.columns)],
                         columns=['display'] + gorilla_df.columns.tolist())
practice_filenames = [
    'top-left.jpg','top-left.jpg','top-right.jpg','top-right.jpg',
    'bottom-left.jpg','bottom-left.jpg','bottom-right.jpg', 'bottom-right.jpg',
]
practice_answers = [1,1,2,2,3,3,4,4]
practice_rows = pd.DataFrame(
    [['Practice'] + [''] * len(gorilla_df.columns) for _ in practice_filenames],
    columns=['display'] + gorilla_df.columns.tolist()
)
practice_rows['filename'] = practice_filenames
practice_rows['answer'] = practice_answers
begin_row = pd.DataFrame([['Begin'] + [''] * len(gorilla_df.columns)],
                         columns=['display'] + gorilla_df.columns.tolist())
gorilla_df.insert(0,'display','Task')
combined_df = pd.concat(
    [intro_row, practice_rows,begin_row, gorilla_df],
    ignore_index=True
)
combined_df.to_csv(os.path.join(new_stim_path, "human_stimuli_gorilla.csv"), index=False)
