import os
import csv
import json
import pandas as pd
from tqdm import tqdm
import re
import math



def process_batch_responses(dataset_dir, annotations_file, results_file, batch_responses_file, model, expect_coordinates=False, rows_and_columns=False):
    # Read the annotations
    if expect_coordinates and rows_and_columns:
        raise ValueError("Cannot have both coordinates and rows and columns")

    annotations = pd.read_csv(os.path.join(dataset_dir, annotations_file))

    # Prepare the results CSV
    if expect_coordinates:
        fieldnames = [
            'filename', 'num_distractors', 'size', 'colourbin',
            'selected_x', 'selected_y',
            'actual_center_x', 'actual_center_y',
            'error_x', 'error_y', 'euclidean_error',
            'selected_response'
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
                custom_id+=".png"

            if model =="gpt-4o":
                response_data = response_entry.get('response')
            elif model == "claude-sonnet":
                response_data = response_entry.get("result")
            elif model=="llama11B" or model=="llama90B":
                response_data=response_entry.get("content")
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
                    
                    elif model =="claude-sonnet":
                        #print(response_data["message"]["content"][0]["text"])
                        assistant_message = response_data['message']['content'][0]["text"]

                    elif model == "llama11B" or "llama90B":
                        assistant_message = response_data
                        
                    selected_response = assistant_message

                    if expect_coordinates:
                        # Coordinate mode
                        # Use regular expressions to extract coordinates from assistant_message
                        coordinate_patterns = [
                            r'[\(\[]\s*([\d.]+)[,\s]+([\d.]+)\s*[\)\]]',  # Matches (x, y) or [x, y]
                            r'Coordinates?:?\s*([\d.]+)[,\s]+([\d.]+)',    # Matches 'Coordinates: x, y'
                            r'x[:=]\s*([\d.]+)[,\s]+y[:=]\s*([\d.]+)',     # Matches 'x: x_value y: y_value'
                            r'[\s]*([\d.]+)[,\s]+([\d.]+)[\s]*'            # Matches 'x y' or 'x, y'
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
                            # Could not parse coordinates
                            error_x = None
                            error_y = None
                            euclidean_error = None
                            actual_center_x = None
                            actual_center_y = None
                        else:
                            # Get the actual center coordinates from annotations
                            target_annotations = annotations[(annotations["filename"] == filename) & (annotations["target"] == True)]

                            if not target_annotations.empty:
                                actual_center_x = target_annotations['center_x'].iloc[0]
                                actual_center_y = target_annotations['center_y'].iloc[0]

                                # Calculate errors
                                error_x = selected_x - actual_center_x
                                error_y = selected_y - actual_center_y
                                euclidean_error = math.sqrt(error_x**2 + error_y**2)
                            else:
                                actual_center_x = None
                                actual_center_y = None
                                error_x = None
                                error_y = None
                                euclidean_error = None

                    elif rows_and_columns:
                        # Rows and Columns mode
                        # Use regular expressions to extract (i, j) from assistant_message
                        coordinate_patterns = [
                            r'[\(\[]\s*(\d+)[,\s]+(\d+)\s*[\)\]]',  # Matches (i, j) or [i, j]
                            r'Row[:=]?\s*(\d+)[,\s]+Column[:=]?\s*(\d+)',  # Matches 'Row: i, Column: j'
                            r'[\s]*Row\s*(\d+)[,\s]+Column\s*(\d+)[\s]*',  # Matches 'Row i, Column j'
                            r'[\s]*(\d+)[,\s]+(\d+)[\s]*'  # Matches 'i j' or 'i, j'
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
                            # Could not parse row and column
                            correct = False
                            actual_row = None
                            actual_col = None
                        else:
                            # Get the actual row and column from annotations
                            target_annotations = annotations[(annotations["filename"] == filename) & (annotations["target"] == True)]

                            if not target_annotations.empty:
                                actual_row = target_annotations['row'].iloc[0]
                                actual_col = target_annotations['column'].iloc[0]

                                # Determine if the prediction is correct
                                correct = (selected_row == actual_row) and (selected_col == actual_col)
                            else:
                                actual_row = None
                                actual_col = None
                                correct = False

                    else:
                        # Quadrant mode
                        # Possible quadrant labels
                        quadrants = annotations['quadrant'].unique().tolist()

                        # Check if any quadrant label is in GPT-4o's response
                        selected_quadrant = 'Unknown'
                        for q in quadrants:
                            if q.lower() in assistant_message.lower():
                                selected_quadrant = q
                                break

                        # Get the actual quadrant from annotations
                        target_annotations = annotations[(annotations["filename"] == filename) & (annotations["target"] == True)]
                        if not target_annotations.empty:
                            actual_quadrant = target_annotations['quadrant'].iloc[0]
                            correct = selected_quadrant == actual_quadrant
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
                selected_response = "No response data available"
                if expect_coordinates:
                    selected_x = None
                    slected_y = None
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

            # Retrieve common annotations for the image
            try:
                # Get common details from annotations
                actual_annotation = annotations[(annotations["filename"] == filename) & (annotations["target"] == True)]
                if not actual_annotation.empty:
                    num_distractors = actual_annotation['num_distractors'].iloc[0]
                    object_size = actual_annotation['size'].iloc[0]
                    colourbin = actual_annotation['color_bin_index'].iloc[0]
                else:
                    num_distractors = ''
                    object_size = ''
                    colourbin = ''
                    print(f"No annotations found for filename: {filename}")
            except Exception as e:
                num_distractors = ''
                object_size = ''
                colourbin = ''
                print(f"Error retrieving annotations for custom_id {custom_id}: {e}")
                continue  # Skip this entry if annotations can't be retrieved

            # Write the result to CSV
            if expect_coordinates:
                writer.writerow({
                    'filename': filename,
                    'num_distractors': num_distractors,
                    'size': object_size,
                    'colourbin': colourbin,
                    'selected_x': selected_x,
                    'selected_y': selected_y,
                    'actual_center_x': actual_center_x,
                    'actual_center_y': actual_center_y,
                    'error_x': error_x,
                    'error_y': error_y,
                    'euclidean_error': euclidean_error,
                    'selected_response': selected_response
                })
            elif rows_and_columns:
                writer.writerow({
                    'filename': filename,
                    'num_distractors': num_distractors,
                    'size': object_size,
                    'colourbin': colourbin,
                    'selected_cell': f"({selected_row}, {selected_col})" if selected_row is not None else 'Unknown',
                    'actual_cell': f"({actual_row}, {actual_col})" if actual_row is not None else 'Unknown',
                    'correct': correct,
                    'selected_response': selected_response
                })
            else:
                writer.writerow({
                    'filename': filename,
                    'num_distractors': num_distractors,
                    'size': object_size,
                    'colourbin': colourbin,
                    'selected_quadrant': selected_quadrant,
                    'actual_quadrant': actual_quadrant,
                    'correct': correct,
                    'selected_response': selected_response
                })

    print(f"Results saved to '{results_file}'.")

# Main script
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directory", required=True)
parser.add_argument("-a", "--annotations_file", default="annotations.csv")
parser.add_argument("-b", "--batch_responses", default="combined_batch_responses.jsonl")
parser.add_argument("-c", "--expect_coords", action='store_true')
parser.add_argument("-rc", "--rowsColumns", action='store_true')
parser.add_argument("-q", "--quadrants", action="store_true")
parser.add_argument("-m", "--model", choices={"gpt-4o", "claude-sonnet", "llama11B", "llama90B"}, required=True)
args = parser.parse_args()

mapping = [
    (args.expect_coords, "Coords"),
    (args.rowsColumns, "Cells"),
    (args.quadrants, "Quadrant")
]
resultFileType = next((value for condition, value in mapping if condition), None)
if resultFileType is None:
    raise ValueError("At least one of -c, -rc, -q must be used!")

# Call the function with the parsed arguments
process_batch_responses(
    dataset_dir="results/"+args.directory,
    annotations_file=args.annotations_file,
    results_file=args.model+"_results_"+resultFileType+".csv",
    batch_responses_file=args.model+"_"+args.batch_responses,
    expect_coordinates=args.expect_coords,
    rows_and_columns=args.rowsColumns,
    model=args.model
)
