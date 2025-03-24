import os
import csv
import json
import pandas as pd
from tqdm import tqdm
import re
import math
import argparse
from collections import defaultdict


def process_batch_responses(dataset_dir, annotations_file, results_file,
                            batch_responses_file, model,
                            expect_coordinates=False,
                            rows_and_columns=False,
                            presence=False):
    # Read the annotations
    if expect_coordinates and rows_and_columns:
        raise ValueError("Cannot have both coordinates and rows and columns")

    annotations = pd.read_csv(os.path.join(dataset_dir, annotations_file))

    # -- BUILD A DICTIONARY FOR FAST LOOKUPS --
    # Exactly one row for each filename has target == True (unless not present at all).
    # We'll store that row in a dictionary for quick O(1) lookups.
    target_rows = {}
    for _, row in annotations.iterrows():
        if row["target"] == True:
            fname = row["filename"]
            target_rows[fname] = row

    # Prepare the results CSV
    if expect_coordinates:
        fieldnames = [
            'filename', 'num_distractors', 'size', 'colourbin',
            'selected_x', 'selected_y',
            'actual_center_x', 'actual_center_y',
            'error_x', 'error_y', 'euclidean_error',
            'selected_response'
        ]
    elif presence:
        fieldnames = [
            'filename', 'num_distractors', 'size', 'colourbin',
            'selected_presence', 'actual_presence', 'selected_response'
        ]
    elif rows_and_columns:
        fieldnames = [
            'filename', 'num_distractors', 'size', 'colourbin',
            'selected_cell', 'actual_cell', 'correct', 'selected_response'
        ]
    else:
        fieldnames = [
            'filename', 'num_distractors', 'size', 'colourbin',
            'selected_quadrant', 'actual_quadrant', 'correct', 'selected_response'
        ]

    with open(os.path.join(dataset_dir, results_file), mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Read and process the batch responses line by line
        with open(os.path.join(dataset_dir, batch_responses_file), 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc='Processing batch responses'):
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            try:
                response_entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {e}")
                continue

            # Extract the custom_id and response data
            custom_id = response_entry.get('custom_id')
            if model == "claude-sonnet":
                custom_id += ".png"

            if model == "gpt-4o":
                response_data = response_entry.get('response')
            elif model == "claude-sonnet":
                response_data = response_entry.get("result")
            elif model in {"llama11B", "llama90B"}:
                response_data = response_entry.get("content")
            else:
                response_data = None

            error = response_entry.get('error')

            # Initialize variables
            selected_response = ''
            filename = custom_id

            # Handle errors
            if error is not None:
                selected_response = f"Error: {error}"
                if expect_coordinates:
                    selected_x = None
                    selected_y = None
                    error_x = None
                    error_y = None
                    euclidean_error = None
                    actual_center_x = None
                    actual_center_y = None
                elif rows_and_columns:
                    selected_cell = 'Error'
                    correct = False
                    actual_row = None
                    actual_col = None
                else:
                    selected_quadrant = 'Error'
                    correct = False
            elif response_data is not None:
                # Extract LLM's response
                try:
                    if model == "gpt-4o":
                        assistant_message = response_data['body']['choices'][0]['message']['content'].strip()
                    elif model == "claude-sonnet":
                        assistant_message = response_data['message']['content'][0]["text"]
                    elif model in {"llama11B", "llama90B"}:
                        assistant_message = response_data

                    selected_response = assistant_message

                    if expect_coordinates:
                        # Coordinate mode
                        coordinate_patterns = [
                            r'[\(\[]\s*([\d.]+)[,\s]+([\d.]+)\s*[\)\]]',  # (x, y) or [x, y]
                            r'Coordinates?:?\s*([\d.]+)[,\s]+([\d.]+)',    # 'Coordinates: x, y'
                            r'x[:=]\s*([\d.]+)[,\s]+y[:=]\s*([\d.]+)',     # 'x: x_value y: y_value'
                            r'[\s]*([\d.]+)[,\s]+([\d.]+)[\s]*'            # 'x y' or 'x, y'
                        ]

                        selected_x = None
                        selected_y = None

                        for pattern in coordinate_patterns:
                            match = re.search(pattern, assistant_message, re.IGNORECASE)
                            if match:
                                selected_x = float(match.group(1))
                                selected_y = float(match.group(2))
                                break

                        if selected_x is None or selected_y is None:
                            error_x = None
                            error_y = None
                            euclidean_error = None
                            actual_center_x = None
                            actual_center_y = None
                        else:
                            # Look up the target row from our dictionary
                            target_row = target_rows.get(filename, None)
                            if target_row is not None:
                                actual_center_x = target_row['center_x']
                                actual_center_y = target_row['center_y']
                                error_x = selected_x - actual_center_x
                                error_y = selected_y - actual_center_y
                                euclidean_error = math.sqrt(error_x**2 + error_y**2)
                            else:
                                # No row for this file
                                actual_center_x = None
                                actual_center_y = None
                                error_x = None
                                error_y = None
                                euclidean_error = None

                    elif rows_and_columns:
                        # Rows and Columns mode
                        coordinate_patterns = [
                            r'[\(\[]\s*(\d+)[,\s]+(\d+)\s*[\)\]]',       # (i, j) or [i, j]
                            r'Row[:=]?\s*(\d+)[,\s]+Column[:=]?\s*(\d+)', # 'Row: i, Column: j'
                            r'[\s]*Row\s*(\d+)[,\s]+Column\s*(\d+)[\s]*', # 'Row i, Column j'
                            r'[\s]*(\d+)[,\s]+(\d+)[\s]*'                # 'i j' or 'i, j'
                        ]

                        selected_row = None
                        selected_col = None

                        for pattern in coordinate_patterns:
                            match = re.search(pattern, assistant_message, re.IGNORECASE)
                            if match:
                                selected_row = int(match.group(1))
                                selected_col = int(match.group(2))
                                break

                        if selected_row is None or selected_col is None:
                            correct = False
                            actual_row = None
                            actual_col = None
                        else:
                            # Look up the target row
                            target_row = target_rows.get(filename, None)
                            if target_row is not None:
                                actual_row = target_row['row']
                                actual_col = target_row['column']
                                correct = (selected_row == actual_row) and (selected_col == actual_col)
                            else:
                                actual_row = None
                                actual_col = None
                                correct = False

                    elif presence:
                        # Presence mode
                        try:
                            # Attempt to parse an integer 0/1
                            selected_presence = int(assistant_message)
                            if selected_presence not in [0, 1]:
                                raise ValueError("Invalid presence value")
                        except Exception as e:
                            selected_response = f"Error extracting response: {e}"
                            selected_presence = "Error"

                    else:
                        # Quadrant mode
                        # Possible quadrant labels
                        quadrants = annotations['quadrant'].unique().tolist()

                        selected_quadrant = 'Unknown'
                        for q in quadrants:
                            if q.lower() in assistant_message.lower():
                                selected_quadrant = q
                                break

                        target_row = target_rows.get(filename, None)
                        if target_row is not None:
                            actual_quadrant = target_row['quadrant']
                            correct = (selected_quadrant == actual_quadrant)
                        else:
                            actual_quadrant = None
                            correct = False

                except Exception as e:
                    selected_response = f"Error extracting response: {e}"
                    if expect_coordinates:
                        selected_x = None
                        selected_y = None
                        error_x = None
                        error_y = None
                        euclidean_error = None
                        actual_center_x = None
                        actual_center_y = None
                    elif rows_and_columns:
                        selected_row = None
                        selected_col = None
                        actual_row = None
                        actual_col = None
                        correct = False
                    else:
                        selected_quadrant = 'Error'
                        actual_quadrant = None
                        correct = False

            else:
                # No response
                selected_response = "No response data available"
                if expect_coordinates:
                    selected_x = None
                    selected_y = None
                    error_x = None
                    error_y = None
                    euclidean_error = None
                    actual_center_x = None
                    actual_center_y = None
                elif rows_and_columns:
                    selected_row = None
                    selected_col = None
                    actual_row = None
                    actual_col = None
                    correct = False
                else:
                    selected_quadrant = 'Error'
                    actual_quadrant = None
                    correct = False

            # Retrieve common annotations for the image (num_distractors, size, colourbin, presence, etc.)
            try:
                target_row = target_rows.get(filename, None)
                if target_row is not None:
                    num_distractors = target_row['num_distractors']
                    object_size = target_row['size']
                    colourbin = target_row['color_bin_index']

                    if presence:
                        # If center_x == -1 => target not present
                        actual_center_x = target_row['center_x']
                        actual_presence = 0 if actual_center_x == -1 else 1
                else:
                    # No row with target==True. We still might want placeholders.
                    num_distractors = ''
                    object_size = ''
                    colourbin = ''
                    if presence:
                        # If there's no row at all, treat it as not present
                        actual_presence = 0

                    print(f"No annotations found for filename: {filename}")

            except Exception as e:
                # If something goes wrong getting the row data
                num_distractors = ''
                object_size = ''
                colourbin = ''
                if presence:
                    actual_presence = ''
                print(f"Error retrieving annotations for custom_id {custom_id}: {e}")
                continue  # Skip writing if annotations can't be retrieved

            # Write the result to CSV
            if expect_coordinates:
                writer.writerow({
                    'filename': filename,
                    'num_distractors': num_distractors,
                    'size': object_size,
                    'colourbin': colourbin,
                    'selected_x': selected_x,
                    'selected_y': selected_y,
                    'actual_center_x': (actual_center_x if target_row is not None else None),
                    'actual_center_y': (actual_center_y if target_row is not None else None),
                    'error_x': (error_x if target_row is not None else None),
                    'error_y': (error_y if target_row is not None else None),
                    'euclidean_error': (euclidean_error if target_row is not None else None),
                    'selected_response': selected_response
                })
            elif rows_and_columns:
                writer.writerow({
                    'filename': filename,
                    'num_distractors': num_distractors,
                    'size': object_size,
                    'colourbin': colourbin,
                    'selected_cell': f"({selected_row}, {selected_col})" if (selected_row is not None and selected_col is not None) else 'Unknown',
                    'actual_cell': f"({actual_row}, {actual_col})" if (actual_row is not None and actual_col is not None) else 'Unknown',
                    'correct': correct,
                    'selected_response': selected_response
                })
            elif presence:
                writer.writerow({
                    'filename': filename,
                    'num_distractors': num_distractors,
                    'size': object_size,
                    'colourbin': colourbin,
                    'selected_presence': selected_presence,
                    'actual_presence': actual_presence,
                    'selected_response': selected_response
                })
            else:
                writer.writerow({
                    'filename': filename,
                    'num_distractors': num_distractors,
                    'size': object_size,
                    'colourbin': colourbin,
                    'selected_quadrant': selected_quadrant if 'selected_quadrant' in locals() else 'Unknown',
                    'actual_quadrant': actual_quadrant if 'actual_quadrant' in locals() else None,
                    'correct': correct if 'correct' in locals() else False,
                    'selected_response': selected_response
                })

    print(f"Results saved to '{results_file}'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", required=True)
    parser.add_argument("-a", "--annotations_file", default="annotations.csv")
    parser.add_argument("-b", "--batch_responses", default="combined_batch_responses.jsonl")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-c", "--expect_coords", action='store_true', help="Coordinate mode")
    group.add_argument("-rc", "--rowsColumns", action='store_true', help="Rows and Columns mode")
    group.add_argument("-q", "--quadrants", action="store_true", help="Quadrant mode")
    group.add_argument("-p", "--presence", action="store_true", help="Presence mode")
    parser.add_argument("-m", "--model", choices={"gpt-4o", "claude-sonnet", "llama11B", "llama90B"}, required=True)
    args = parser.parse_args()

    # Determine what kind of result file we need
    mode_mapping = [
        (args.presence, "Presence"),
        (args.expect_coords, "Coords"),
        (args.rowsColumns, "Cells"),
        (args.quadrants, "Quadrant")
    ]
    resultFileType = next((val for cond, val in mode_mapping if cond), None)
    if resultFileType is None:
        raise ValueError("At least one of -c, -rc, -q, -p must be used!")

    process_batch_responses(
        dataset_dir="results/" + args.directory,
        annotations_file=args.annotations_file,
        results_file=f"{args.model}_results_{resultFileType}.csv",
        batch_responses_file=f"{args.model}_{args.batch_responses}",
        expect_coordinates=args.expect_coords,
        rows_and_columns=args.rowsColumns,
        presence=args.presence,
        model=args.model
    )


if __name__ == "__main__":
    main()
