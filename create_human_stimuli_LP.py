import os
import pandas as pd
import numpy as np
from PIL import Image
import random
from collections import Counter

path = os.getcwd()
stim_path = os.path.join(path, "original_stimuli")
new_stim_path = os.path.join(path, "human_stimuli")

# Experiment parameters
def sample_from_bins(bins, seed=None, samples_per_bin=4):
    """Sample specified number of items from each bin"""
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
    range(9, 13),  # 9 to 12
    range(13, 17),  # 13 to 16
    range(17, 21),  # 17 to 20
    range(21, 25),  # 21 to 24
    range(25, 33),  # 25 to 32
    range(33, 50),  # 33 to 49
]
samples_per_bin = 4
distractor_nums = sample_from_bins(
    distractor_bins,
    seed=random_seed,
    samples_per_bin=samples_per_bin
)
types_of_stimuli = [
    "TopAmongBottom",
    "BottomAmongTop",
    "LeftAmongRight",
    "RightAmongLeft"
]
quadrants = [
    "Quadrant 1",
    "Quadrant 2",
    "Quadrant 3",
    "Quadrant 4"
]

def get_bin_index(num, bins):
    """Get the index of the bin containing the given number"""
    for i, bin_range in enumerate(bins):
        if num in bin_range:
            return i
    return None


def quadrant_sample(data, distractor_nums, distractor_bins):
    """
    Select stimuli ensuring each bin has exactly one stimulus from each quadrant
    Try to maximize the variety of distractor numbers used
    """
    data = data.copy()
    result = pd.DataFrame()
    rng = random.Random(random_seed)
    distractor_to_bin = {num: get_bin_index(num, distractor_bins) for num in distractor_nums}
    bins_to_distractor_nums = {}
    for num in distractor_nums:
        bin_idx = distractor_to_bin[num]
        if bin_idx not in bins_to_distractor_nums:
            bins_to_distractor_nums[bin_idx] = []
        bins_to_distractor_nums[bin_idx].append(num)
    selected_info = {}
    for bin_idx, nums_in_bin in bins_to_distractor_nums.items():
        bin_data = data[data["num_distractors"].isin(nums_in_bin)]
        selected_info[bin_idx] = {"quadrants": {}, "numbers": []}
        random_quadrant_order = quadrants.copy()
        rng.shuffle(random_quadrant_order)
        all_available_nums = sorted(bin_data["num_distractors"].unique())
        if len(all_available_nums) >= 4:
            selected_distractor_nums = rng.sample(all_available_nums, 4)
        else:
            selected_distractor_nums = all_available_nums
            while len(selected_distractor_nums) < 4:
                selected_distractor_nums.append(rng.choice(all_available_nums))
        for i, quadrant in enumerate(random_quadrant_order):
            selected_num = selected_distractor_nums[i]
            quadrant_specific_data = bin_data[(bin_data["num_distractors"] == selected_num) &
                                              (bin_data["quadrant"] == quadrant)]
            if quadrant_specific_data.empty:
                available_quadrants = bin_data[bin_data["num_distractors"] == selected_num]["quadrant"].unique()
                if len(available_quadrants) > 0:
                    alternative_quadrant = rng.choice(available_quadrants)
                    quadrant_specific_data = bin_data[(bin_data["num_distractors"] == selected_num) &
                                                      (bin_data["quadrant"] == alternative_quadrant)]
                    print(
                        f"Warning: Swapped quadrant {quadrant} to {alternative_quadrant} for distractor {selected_num} in bin {bin_idx}")
                else:
                    alternative_numbers = bin_data[bin_data["quadrant"] == quadrant]["num_distractors"].unique()
                    if len(alternative_numbers) > 0:
                        selected_num = rng.choice(alternative_numbers)
                        quadrant_specific_data = bin_data[(bin_data["num_distractors"] == selected_num) &
                                                          (bin_data["quadrant"] == quadrant)]
                        print(
                            f"Warning: Changed distractor from {selected_distractor_nums[i]} to {selected_num} for quadrant {quadrant} in bin {bin_idx}")
                    else:
                        print(f"Warning: No data for quadrant {quadrant} in bin {bin_idx}")
                        continue
            selected_info[bin_idx]["quadrants"][quadrant] = True
            if selected_num not in selected_info[bin_idx]["numbers"]:
                selected_info[bin_idx]["numbers"].append(selected_num)
            selected_row = quadrant_specific_data.sample(n=1, random_state=rng.randint(0, 10000))
            result = pd.concat([result, selected_row], ignore_index=True)
    for bin_idx in selected_info.keys():
        bin_range = f"{list(distractor_bins[bin_idx])[0]}-{list(distractor_bins[bin_idx])[-1]}"
        print(f"\nBin {bin_idx} ({bin_range}):")
        bin_nums = bins_to_distractor_nums[bin_idx]
        bin_result = result[result["num_distractors"].isin(bin_nums)]
        quadrant_dist = bin_result["quadrant"].value_counts().to_dict()
        print(f"  Quadrant distribution: {quadrant_dist}")
        print(f"  Selected distractor numbers: {sorted(selected_info[bin_idx]['numbers'])}")
    return result

# sample from stimuli
cols = ['filename', 'shape_type', 'center_x', 'center_y', 'rotation_angle', 'color', 'quadrant', 'num_distractors']
new_df = pd.DataFrame()
for st in types_of_stimuli:
    directory = os.path.join(stim_path, st)
    df = pd.read_csv(os.path.join(directory, "annotations.csv"))
    df = df[df['target'] == True][cols].copy()
    df.rename(columns={'color': 'target_color'}, inplace=True)
    df = df[~((df['center_x'].between(170, 230)) |
              (df['center_y'].between(170, 230)))]
    df["type"] = st
    df = df[df["num_distractors"].isin(distractor_nums)]
    temp_df = quadrant_sample(
        data=df,
        distractor_nums=distractor_nums,
        distractor_bins=distractor_bins
    )
    new_df = pd.concat([new_df, temp_df], ignore_index=True)

# Print summary
print("\nSummary of selected stimuli:")
print(f"Total stimuli selected: {len(new_df)}")
print(f"Stimuli by type: {new_df['type'].value_counts().to_dict()}")

## load and save images
for i, file in enumerate(new_df["filename"]):
    directory = os.path.join(stim_path, new_df.loc[i, "type"])
    img = Image.open(os.path.join(directory, file))
    img_name = f"image_{i}.jpg"
    new_df.loc[i, "filename_new"] = img_name
    img.convert("RGB").save(os.path.join(new_stim_path, img_name))
new_df.to_csv(os.path.join(new_stim_path, "human_stimuli.csv"), index=False)

## format df for gorilla
gradient_dict = {
    "TopAmongBottom":"vertical",
    "BottomAmongTop":"vertical",
    "LeftAmongRight":"horizontal",
    "RightAmongLeft":"horizontal",
}
inversion_dict = {
    "TopAmongBottom":"original",
    "BottomAmongTop":"inverted",
    "LeftAmongRight":"original",
    "RightAmongLeft":"inverted",
}
light_dict = {
    "TopAmongBottom":"top",
    "BottomAmongTop":"bottom",
    "LeftAmongRight":"left",
    "RightAmongLeft":"right",
}

gorilla_cols = ['filename','gradient_type','inversion_type','light_direction','quadrant','num_distractors','answer']
gorilla_df = new_df.copy()
gorilla_df.drop(labels=["filename"], axis=1, inplace=True)
gorilla_df.rename(columns={'filename_new': 'filename'}, inplace=True)
gorilla_df['answer'] = gorilla_df['quadrant'].str[-1].astype(int)
gorilla_df['gradient_type'] = gorilla_df['type'].replace(gradient_dict)
gorilla_df['inversion_type'] = gorilla_df['type'].replace(inversion_dict)
gorilla_df['light_direction'] = gorilla_df['type'].replace(light_dict)
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