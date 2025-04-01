import argparse
import random
import csv
import os
import time

from PIL import Image, ImageDraw, ImageColor

def generate_images(dir, num_images, min_k, max_k, c, targetShape, distractorShape, 
                    shapeSize, theta_min, theta_max, targetColour, distractorColour, conjunctive=False,
                    grid_rows=2, grid_cols=2, quadrantOrder=None, debug=False, present=False, colourMode="explicit", colourList=None):
    # Set image dimensions
    width, height = 400, 400  # You can adjust the size as needed


    # Generate quadrants based on grid dimensions and quadrantOrder
    num_quadrants = grid_rows * grid_cols

    # Generate quadrants based on grid dimensions and quadrantOrder
    if quadrantOrder is None:
        # Default quadrant labels in order
        quadrants = [f"Quadrant {i+1}" for i in range(num_quadrants)]
    else:
        if len(quadrantOrder) != num_quadrants:
            raise ValueError("quadrantOrder length does not match number of quadrants")
        # Use the provided quadrantOrder for labeling
        quadrants = [f"Quadrant {q}" for q in quadrantOrder]

    # Output directory for images
    if args.finetuning:
        output_dir = "finetuning/"+dir
    else:
        output_dir = "results/"+dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open a CSV file to save the annotations
    with open(os.path.join(output_dir, 'annotations.csv'), mode='w', newline='') as csv_file:
        fieldnames = ['filename', 'shape_type', 'target', 'center_x', 'center_y', 'size',
                      'color', 'quadrant', 'row', 'column', 'num_distractors', 'num_images', 'distractor_color',
                      'color_bin_index', 'rotation_angle']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # Grid cell dimensions
        col_width = width / grid_cols
        row_height = height / grid_rows

        for i in range(num_images):



            if colourMode == "explicit":
                currentTargetColour = targetColour
                currentDistractorColour = distractorColour
            elif colourMode == "randomDifferent":
                if not colourList or len(colourList) < 2:
                    raise ValueError("randomDifferent mode requires at least two colours in --colourList")
                currentTargetColour, currentDistractorColour = random.sample(colourList, 2)
            elif colourMode == "randomSame":
                if not colourList or len(colourList) < 1:
                    raise ValueError("randomSame mode requires at least one colour in --colourList")
                chosenColor = random.choice(colourList)
                currentTargetColour = chosenColor
                currentDistractorColour = chosenColor




            if present:
                targetPresent = (random.random()<0.5)
            else:
                targetPresent = True



            # Randomly select k independently
            k = random.randint(min_k, max_k)



            # Create a new image with a white background
            image = Image.new('RGBA', (width, height), 'white')

            # If debug is True, draw the grid lines
            if debug:
                image_draw = ImageDraw.Draw(image)
                # Draw vertical grid lines
                for col in range(1, grid_cols):
                    x = col * col_width
                    image_draw.line([(x, 0), (x, height)], fill='green', width=1)
                # Draw horizontal grid lines
                for row in range(1, grid_rows):
                    y = row * row_height
                    image_draw.line([(0, y), (width, y)], fill='green', width=1)

            occupied_areas = []

            # Adjust the canvas size to accommodate the rotated shape
            padding = shapeSize * 0.05  # Existing padding
            canvas_size = shapeSize + 2*padding

            # Generate random position for the target's center within the main image
            max_center_x = width - canvas_size / 2
            max_center_y = height - canvas_size / 2
            min_center_x = canvas_size / 2
            min_center_y = canvas_size / 2

            if targetPresent:

                # Random rotation angle for the target
                target_rotation = random.uniform(theta_min, theta_max)

                

                target_center_x = random.uniform(min_center_x, max_center_x)
                target_center_y = random.uniform(min_center_y, max_center_y)

                # Determine target color
                if c == -1:
                    # Random color from red, green, or blue
                    target_color_options = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
                    target_color = random.choice(target_color_options)
                    target_color_hex = '#%02x%02x%02x' % target_color
                    target_color_bin_index = None  # Not applicable when c == -1
                elif c == -2:
                    # Target color is distractorColour
                    target_color = ImageColor.getrgb(distractorColour)
                    target_color_hex = distractorColour
                    target_color_bin_index = 0  # Since the target is the same as distractor
                else:
                    # Target color
                    target_color = ImageColor.getrgb(currentTargetColour)
                    target_color_hex = currentTargetColour
                    target_color_bin_index = 0  # Since the target is targetColour

                # Draw the target shape and get the image
                target_shape_image = draw_shape(canvas_size, shapeSize, targetShape, target_color)

                # Rotate the target shape without expanding the canvas
                target_shape_image = target_shape_image.rotate(-target_rotation, expand=False, resample=Image.BICUBIC)

                # Adjust position to keep shape within image boundaries
                adjusted_target_x = target_center_x - canvas_size / 2
                adjusted_target_y = target_center_y - canvas_size / 2

                # Ensure the shape is within the image boundaries
                adjusted_target_x = min(max(adjusted_target_x, 0), width - canvas_size)
                adjusted_target_y = min(max(adjusted_target_y, 0), height - canvas_size)

                # Update the center coordinates after adjustments
                target_center_x = adjusted_target_x + canvas_size / 2
                target_center_y = adjusted_target_y + canvas_size / 2

                # Paste the rotated shape onto the main image
                image.paste(target_shape_image, (int(adjusted_target_x), int(adjusted_target_y)), target_shape_image)

                # If debug is True, draw the bounding box and center cross around the target
                if debug:
                    image_draw = ImageDraw.Draw(image)
                    bbox_draw = [adjusted_target_x, adjusted_target_y, adjusted_target_x + canvas_size, adjusted_target_y + canvas_size]
                    image_draw.rectangle(bbox_draw, outline='red', width=2)
                    # Draw a small cross at the center
                    cross_size = 5  # Length of the cross arms
                    image_draw.line(
                        [
                            (target_center_x - cross_size, target_center_y),
                            (target_center_x + cross_size, target_center_y)
                        ],
                        fill='blue', width=1
                    )
                    image_draw.line(
                        [
                            (target_center_x, target_center_y - cross_size),
                            (target_center_x, target_center_y + cross_size)
                        ],
                        fill='blue', width=1
                    )

                # Add target to occupied areas
                occupied_areas.append({
                    'type': targetShape,
                    'x': adjusted_target_x,
                    'y': adjusted_target_y,
                    'size': canvas_size  # Use canvas_size for width and height
                })

                # Determine the quadrant for the target
                col_index = int(target_center_x / col_width)
                row_index = int(target_center_y / row_height)
                # Ensure indices are within bounds
                col_index = min(col_index, grid_cols - 1)
                row_index = min(row_index, grid_rows - 1)
                # Calculate the quadrant index
                quadrant_index = row_index * grid_cols + col_index
                quadrant = quadrants[quadrant_index]
                row_number=row_index+1
                column_number=col_index+1
                # Save the image filename
                filename = f'image_{i}.png'

                # Write the annotation for the target
                writer.writerow({
                    'filename': filename,
                    'shape_type': targetShape,
                    'target': True,
                    'center_x': target_center_x,
                    'center_y': target_center_y,
                    'size': shapeSize,
                    'color': target_color_hex,
                    'quadrant': quadrant,
                    'row': row_number,
                    'column': column_number,
                    'num_distractors': k,          # Include k in the CSV
                    'num_images': num_images,      # Include num_images in the CSV
                    'distractor_color': 'various' if c == -1 else target_color_hex,
                    'color_bin_index': target_color_bin_index,  # None when c == -1
                    'rotation_angle': target_rotation          # Include rotation angle
                })

            # For c == -2, select one distractor to be targetColour
            if c == -2:
                if k <= 1:
                    special_distractor_index = 0
                else:
                    special_distractor_index = random.randint(0, k - 1)



            if conjunctive:
                # W build a small function that randomly picks a
                # distractor shape–color combo. It can share shape or color
                # with the target, but not both (which would be the actual target).
                possible_shapes = [targetShape, distractorShape]
                possible_colors = [ImageColor.getrgb(currentTargetColour),
                                   ImageColor.getrgb(currentDistractorColour)]
                
                def get_random_multifeature_combo():
                    while True:
                        s = random.choice(possible_shapes)
                        c = random.choice(possible_colors)
                        # Exclude the exact combination = (targetShape, targetColour)
                        if not (s == targetShape and c == ImageColor.getrgb(currentTargetColour)):
                            return s, c



            # Generate k distractors
            for distractor_index in range(k):
                max_attempts = 100  # Prevent infinite loops
                attempt = 0
                while attempt < max_attempts:
                    attempt += 1

                    # Random rotation angle for the distractor
                    distractor_rotation = random.uniform(theta_min, theta_max)

                    # Adjust the canvas size to accommodate the rotated shape
                    canvas_size = shapeSize + 2*padding

                    # Randomly choose position for the distractor's center
                    max_center_x = width - canvas_size / 2
                    max_center_y = height - canvas_size / 2
                    min_center_x = canvas_size / 2
                    min_center_y = canvas_size / 2

                    distractor_center_x = random.uniform(min_center_x, max_center_x)
                    distractor_center_y = random.uniform(min_center_y, max_center_y)



                    if conjunctive:
                        # override shape and color if multiFeature is set.
                        chosenShape, chosenColor = get_random_multifeature_combo()
                        distractor_shape = chosenShape
                        distractor_color = chosenColor
                        distractor_color_hex = '#%02x%02x%02x' % distractor_color
                        distractor_color_bin_index = None  # or some placeholder

                    else:


                        # Determine distractor color
                        if c == -1:
                            # Random color
                            distractor_color = (
                                random.randint(0, 255),
                                random.randint(0, 255),
                                random.randint(0, 255)
                            )
                            distractor_color_hex = '#%02x%02x%02x' % distractor_color
                            distractor_color_bin_index = None  # Not applicable when c == -1
                        elif c == -2:
                            # All distractors are distractorColour except for one that is targetColour
                            if distractor_index == special_distractor_index:
                                distractor_color = ImageColor.getrgb(currentTargetColour)
                                distractor_color_hex = currentTargetColour
                                distractor_color_bin_index = 1  # For special distractor
                            else:
                                distractor_color = ImageColor.getrgb(currentDistractorColour)
                                distractor_color_hex = currentDistractorColour
                                distractor_color_bin_index = 0
                        else:
                            # Use colors based on c
                            if c == 0:
                                # If c is 0, use targetColour as distractor color
                                distractor_color = ImageColor.getrgb(currentTargetColour)
                                distractor_color_hex = currentTargetColour
                                distractor_color_bin_index = 0
                            elif c == 1:
                                # If c is 1, use distractorColour
                                distractor_color = ImageColor.getrgb(currentDistractorColour)
                                distractor_color_hex = currentDistractorColour
                                distractor_color_bin_index = 0  # Only one distractor color
                            else:
                                # Interpolate between targetColour and distractorColour
                                start_color_rgb = ImageColor.getrgb(currentTargetColour)
                                end_color_rgb = ImageColor.getrgb(currentDistractorColour)
                                distractor_colours = []

                                for j in range(c):
                                    fraction = j / (c - 1) if c > 1 else 0
                                    R = round(start_color_rgb[0] * (1 - fraction) + end_color_rgb[0] * fraction)
                                    G = round(start_color_rgb[1] * (1 - fraction) + end_color_rgb[1] * fraction)
                                    B = round(start_color_rgb[2] * (1 - fraction) + end_color_rgb[2] * fraction)
                                    distractor_colours.append((j, (R, G, B)))  # Include color bin index j

                                distractor_color_bin_index, distractor_color = random.choice(distractor_colours)
                                distractor_color_hex = '#%02x%02x%02x' % distractor_color

                    # Draw the distractor shape and get the image
                    distractor_shape_image = draw_shape(canvas_size, shapeSize, distractorShape, distractor_color)

                    # Rotate the distractor shape without expanding the canvas
                    distractor_shape_image = distractor_shape_image.rotate(-distractor_rotation, expand=False, resample=Image.BICUBIC)

                    # Adjust position to keep shape within image boundaries
                    adjusted_distractor_x = distractor_center_x - canvas_size / 2
                    adjusted_distractor_y = distractor_center_y - canvas_size / 2

                    # Ensure the shape is within the image boundaries
                    adjusted_distractor_x = min(max(adjusted_distractor_x, 0), width - canvas_size)
                    adjusted_distractor_y = min(max(adjusted_distractor_y, 0), height - canvas_size)

                    # Update the center coordinates after adjustments
                    distractor_center_x = adjusted_distractor_x + canvas_size / 2
                    distractor_center_y = adjusted_distractor_y + canvas_size / 2

                    # Check for overlaps
                    overlap = False
                    for shape in occupied_areas:
                        if bounding_boxes_overlap(
                            shape['x'], shape['y'], shape['size'], shape['size'],
                            adjusted_distractor_x, adjusted_distractor_y, canvas_size, canvas_size):
                            overlap = True
                            break

                    if not overlap:
                        # No overlap, accept the distractor
                        occupied_areas.append({
                            'type': distractorShape,
                            'x': adjusted_distractor_x,
                            'y': adjusted_distractor_y,
                            'size': canvas_size  # Use canvas_size for width and height
                        })
                        break  # Move to next distractor
                    else:
                        continue  # Try another position

                else:
                    # Could not place distractor after max_attempts
                    print(f"Could not place distractor {distractor_index} in image {i} without overlap after {max_attempts} attempts.")
                    continue  # Skip this distractor

                # Paste the rotated distractor onto the main image
                image.paste(distractor_shape_image, (int(adjusted_distractor_x), int(adjusted_distractor_y)), distractor_shape_image)

                # If debug is True, draw the bounding box and center cross around the distractor
                if debug:
                    image_draw = ImageDraw.Draw(image)
                    bbox_draw = [adjusted_distractor_x, adjusted_distractor_y, adjusted_distractor_x + canvas_size, adjusted_distractor_y + canvas_size]
                    image_draw.rectangle(bbox_draw, outline='red', width=2)
                    # Draw a small cross at the center
                    cross_size = 5  # Length of the cross arms
                    image_draw.line(
                        [
                            (distractor_center_x - cross_size, distractor_center_y),
                            (distractor_center_x + cross_size, distractor_center_y)
                        ],
                        fill='blue', width=1
                    )
                    image_draw.line(
                        [
                            (distractor_center_x, distractor_center_y - cross_size),
                            (distractor_center_x, distractor_center_y + cross_size)
                        ],
                        fill='blue', width=1
                    )

                # Determine the quadrant for the distractor
                col_index = int(distractor_center_x / col_width)
                row_index = int(distractor_center_y / row_height)
                # Ensure indices are within bounds
                col_index = min(col_index, grid_cols - 1)
                row_index = min(row_index, grid_rows - 1)
                # Calculate the quadrant index
                quadrant_index = row_index * grid_cols + col_index
                quadrant = quadrants[quadrant_index]

                column_number= col_index+1
                row_number=row_index+1
                # Write the annotation for the distractor
                filename=f'image_{i}.png'
                writer.writerow({
                    'filename': filename,
                    'shape_type': distractorShape,
                    'target': False,
                    'center_x': distractor_center_x,
                    'center_y': distractor_center_y,
                    'size': shapeSize,
                    'color': distractor_color_hex,
                    'quadrant': quadrant,
                    'row': row_number,
                    'column': column_number,
                    'num_distractors': k,               # Include k in the CSV
                    'num_images': '',                   # Empty for distractors
                    'distractor_color': distractor_color_hex,
                    'color_bin_index': distractor_color_bin_index,  # None when c == -1
                    'rotation_angle': distractor_rotation
                })

            # Save the image
            filename = f'image_{i}.png'
            image.save(os.path.join(output_dir, filename))

    print(f"Generated {num_images} images in the '{output_dir}' directory with k ranging from {min_k} to {max_k}, c = {c}, and rotations between {theta_min}° and {theta_max}°.")


def draw_shape(canvas_size, shapeSize, shape, color):
    # Returns an Image containing the shape
    from PIL import Image, ImageDraw
    import math

    # Create an image to draw the shape
    shape_image = Image.new('RGBA', (int(canvas_size), int(canvas_size)), (0, 0, 0, 0))
    half_canvas_size = canvas_size / 2

    if shape == 'circle':
        # Draw a regular circle
        radius = shapeSize * 0.9 / 2
        left_up_point = (half_canvas_size - radius, half_canvas_size - radius)
        right_down_point = (half_canvas_size + radius, half_canvas_size + radius)
        draw = ImageDraw.Draw(shape_image)
        draw.ellipse([left_up_point, right_down_point], fill=color)

    elif shape in ['circle_gradient_top_bottom', 'circle_gradient_bottom_top']:
        # Use the same adjusted size as other shapes
        size_adjusted = shapeSize * 0.9
        radius = size_adjusted / 2

        # Create gradient circle
        gradient_size = int(math.ceil(size_adjusted))
        gradient = Image.new('RGBA', (gradient_size, gradient_size), (0, 0, 0, 0))
        gradient_draw = ImageDraw.Draw(gradient)
        for y in range(gradient_size):
            if shape == 'circle_gradient_top_bottom':
                fraction = y / (gradient_size - 1) if gradient_size > 1 else 0
            else:
                fraction = 1 - (y / (gradient_size - 1) if gradient_size > 1 else 0)
            # Calculate the color at this row to fade to white
            row_color = (
                int(color[0] + (255 - color[0]) * fraction),
                int(color[1] + (255 - color[1]) * fraction),
                int(color[2] + (255 - color[2]) * fraction),
                255  # Full opacity
            )
            gradient_draw.line([(0, y), (gradient_size - 1, y)], fill=row_color)

        # Create a circular mask
        mask = Image.new('L', (gradient_size, gradient_size), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.ellipse([0, 0, gradient_size - 1, gradient_size - 1], fill=255)

        # Apply the mask to the gradient
        gradient.putalpha(mask)

        # Paste the gradient circle onto the shape_image
        position = (int(half_canvas_size - radius), int(half_canvas_size - radius))
        shape_image.paste(gradient, position, gradient)

    
        draw = ImageDraw.Draw(shape_image)
        outline_left_up = position
        outline_right_down = (position[0] + gradient_size - 1, position[1] + gradient_size - 1)
        draw.ellipse([outline_left_up, outline_right_down], outline="black")


    elif shape == 'square':
        # Draw a square centered at (x, y)
        size_adjusted = shapeSize * 0.9
        half_size_adjusted = size_adjusted / 2
        left_up_point = (half_canvas_size - half_size_adjusted, half_canvas_size - half_size_adjusted)
        right_down_point = (half_canvas_size + half_size_adjusted, half_canvas_size + half_size_adjusted)
        draw = ImageDraw.Draw(shape_image)
        draw.rectangle([left_up_point, right_down_point], fill=color)

    elif shape == 'triangle':
        # Draw an equilateral triangle centered at (x, y)
        size_adjusted = shapeSize * 0.9
        height = size_adjusted * (math.sqrt(3) / 2)
        points = [
            (half_canvas_size, half_canvas_size - height / 2),
            (half_canvas_size - size_adjusted / 2, half_canvas_size + height / 2),
            (half_canvas_size + size_adjusted / 2, half_canvas_size + height / 2)
        ]
        draw = ImageDraw.Draw(shape_image)
        draw.polygon(points, fill=color)

    elif shape in ['2', '5']:
        # Draw digits '2' or '5' using the seven-segment representation
        draw = ImageDraw.Draw(shape_image)
        draw_seven_segment_digit(draw, half_canvas_size, half_canvas_size, shapeSize * 0.85, digit=shape, color=color)

    elif shape == 'T':
        draw = ImageDraw.Draw(shape_image)
        # Use a scale factor less than before to give extra margin
        scaleFactor = 0.8  
        offset = (canvas_size - shapeSize * scaleFactor) / 2
        # Thinner bars for the T
        barHeight = shapeSize * 0.2  
        stemWidth = shapeSize * 0.2  
        # Draw the horizontal bar (top of T)
        draw.rectangle([(offset, offset), 
                        (offset + shapeSize * scaleFactor, offset + barHeight)], fill=color)
        # Draw the vertical stem, centered horizontally within the canvas
        stemLeft = (canvas_size - stemWidth) / 2
        stemTop = offset + barHeight
        draw.rectangle([(stemLeft, stemTop), 
                        (stemLeft + stemWidth, offset + shapeSize * scaleFactor)], fill=color)

    elif shape == 'L':
        draw = ImageDraw.Draw(shape_image)
        scaleFactor = 0.8  # Reduced scale factor for a margin
        offset = (canvas_size - shapeSize * scaleFactor) / 2
        # Thinner bar for the L shape
        barThickness = shapeSize * 0.2  
        # Draw the vertical bar on the left
        draw.rectangle([(offset, offset), 
                        (offset + barThickness, offset + shapeSize * scaleFactor)], fill=color)
        # Draw the horizontal bar at the bottom
        draw.rectangle([(offset, offset + shapeSize * scaleFactor - barThickness),
                        (offset + shapeSize * scaleFactor, offset + shapeSize * scaleFactor)], fill=color)


    else:
        raise ValueError(f"Unsupported shape: {shape}")

    return shape_image

def draw_seven_segment_digit(draw, x, y, size, digit, color):
    # Draws a seven-segment digit centered at position (x, y) with the given size
    # Define segment dimensions
    segment_width = size * 0.2
    gap = size * 0.05  # Gap between segments

    # Coordinates for segments relative to the center (x, y)
    half_size = size / 2

    segments = {
        'A': [(-half_size + gap, -half_size), (half_size - gap, -half_size + segment_width)],  # Top
        'B': [(half_size - segment_width, -half_size + gap), (half_size, 0 - gap)],            # Upper-right
        'C': [(half_size - segment_width, 0 + gap), (half_size, half_size - gap)],             # Lower-right
        'D': [(-half_size + gap, half_size - segment_width), (half_size - gap, half_size)],    # Bottom
        'E': [(-half_size, 0 + gap), (-half_size + segment_width, half_size - gap)],           # Lower-left
        'F': [(-half_size, -half_size + gap), (-half_size + segment_width, 0 - gap)],          # Upper-left
        'G': [(-half_size + gap, 0 - segment_width / 2), (half_size - gap, 0 + segment_width / 2)],  # Middle
    }

    # Offset the segments to the center position (x, y)
    for key in segments:
        segments[key] = [
            (x + segments[key][0][0], y + segments[key][0][1]),
            (x + segments[key][1][0], y + segments[key][1][1])
        ]

    # Define which segments are on for each digit
    digits = {
        '2': ['A', 'B', 'G', 'E', 'D'],
        '5': ['A', 'F', 'G', 'C', 'D'],
    }

    # Draw the segments
    for segment in digits[digit]:
        coords = segments[segment]
        draw.rectangle(coords, fill=color)

def bounding_boxes_overlap(x1, y1, w1, h1, x2, y2, w2, h2):
    # Check if two rectangles overlap
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--colour", type=int, default=None, help="Colour mode (use None to keep preset value)")
    parser.add_argument("-d", "--filename", required=True)
    # Set default for distractors to None so we know if it was explicitly provided.
    parser.add_argument("-dn", "--distractors", type=int, default=None, help="Override preset for max_k")
    parser.add_argument("-r", "--rotation", type=int, default=None)
    parser.add_argument("-n", "--number", type=int, default=None)
    parser.add_argument("-z", "--debug", dest="debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-debug", dest="debug", action="store_false", help="Disable debug mode")
    parser.set_defaults(debug=False)
    parser.add_argument("-pr", "--present", action="store_true", help="Chance to have no target")
    parser.add_argument("-t", "--target", default=None)
    parser.add_argument("-di", "--distractor", default=None)
    parser.add_argument("-tc", "--targetColour", default=None)
    parser.add_argument("-dc", "--distractorColour", default=None)
    parser.add_argument("-s", "--size", type=int, default=None)
    parser.add_argument("-q", "--quadrants", type=str, default=None, help="Specify rows,cols")
    parser.add_argument("-qo", "--quadrantOrder", type=str, default=None, help="Comma-separated list of quadrant integers")
    parser.add_argument("-p", "--preset", default=None)
    parser.add_argument("-f", "--finetuning", action="store_true")
    parser.add_argument("--conjunctive", action="store_true", help="Allow distractors to share target’s shape or color")
    parser.add_argument("--seed", type=int, default=int(time.time()), help="Random seed (default: current time)")
    
    # New colour mode options:
    parser.add_argument("--colourMode", type=str, default="explicit",
                        choices=["explicit", "randomDifferent", "randomSame"],
                        help="Colour selection mode")
    parser.add_argument("--colourList", type=str, default=None,
                        help="Comma-separated list of hex colours for random colour modes")

    args = parser.parse_args()


    if args.quadrants is not None:
        try:
            grid_rows_str, grid_cols_str = args.quadrants.split(',')
            grid_rows = int(grid_rows_str)
            grid_cols = int(grid_cols_str)
        except ValueError:
            raise ValueError("Quadrants must be specified as 'rows,cols'")
    else:
        # Default to 2x2 grid if quadrants not specified
        grid_rows = 2
        grid_cols = 2

    if args.quadrantOrder:
        quadrantOrder = [int(q.strip()) for q in args.quadrantOrder.split(',')]
    else:
        quadrantOrder = None

    random.seed(args.seed)
    print(f"Using random seed: {args.seed}")

    
    # Define presets as a dictionary.
    presets = {
        "2Among5Colour": {
            "num_images": args.number if args.number is not None else 1000,
            "min_k": 0,
            "max_k": 99,
            "c": 1,
            "targetShape": "2",
            "distractorShape": "5",
            "shapeSize": 20,
            "theta_min": 0,
            "theta_max": 360,
            "targetColour": "#00FF00",
            "distractorColour": "#0000FF",
            "quadrantOrder": [1, 2, 3, 4],
            "debug": False,
            "present": False,
            "conjunctive": False,
        },
        "conjunctive": {
            "num_images": args.number if args.number is not None else 1000,
            "min_k": 0,
            "max_k": 99,
            "c": 1,
            "targetShape": "2",
            "distractorShape": "5",
            "shapeSize": 20,
            "theta_min": 0,
            "theta_max": 360,
            "targetColour": "#00FF00",
            "distractorColour": "#0000FF",
            "quadrantOrder": [1, 2, 3, 4],
            "debug": False,
            "present": False,
            "conjunctive": True,
        },
        "2Among5ColourDn": {
            "num_images": args.number if args.number is not None else 1000,
            "min_k": 0,
            "max_k": args.distractors if args.distractors is not None else 99,
            "c": 1,
            "targetShape": "2",
            "distractorShape": "5",
            "shapeSize": 20,
            "theta_min": 0,
            "theta_max": 360,
            "targetColour": "#00FF00",
            "distractorColour": "#0000FF",
            "quadrantOrder": [1, 2, 3, 4],
            "debug": False,
            "present": False,
            "conjunctive": False,
        },
        "2Among5ColourPresent": {
            "num_images": args.number if args.number is not None else 1000,
            "min_k": 0,
            "max_k": 99,
            "c": 1,
            "targetShape": "2",
            "distractorShape": "5",
            "shapeSize": 20,
            "theta_min": 0,
            "theta_max": 360,
            "targetColour": "#00FF00",
            "distractorColour": "#0000FF",
            "quadrantOrder": [1, 2, 3, 4],
            "debug": False,
            "present": True,
            "conjunctive": False,
        },
        "2Among5NoColour": {
            "num_images": args.number if args.number is not None else 1000,
            "min_k": 0,
            "max_k": 99,
            "c": 0,
            "targetShape": "2",
            "distractorShape": "5",
            "shapeSize": 20,
            "theta_min": 0,
            "theta_max": 360,
            "targetColour": "#00FF00",
            "distractorColour": "",
            "quadrantOrder": [1, 2, 3, 4],
            "debug": False,
            "present": False,
            "conjunctive": False,
        },
        "2Among5NoColourPresent": {
            "num_images": args.number if args.number is not None else 1000,
            "min_k": 0,
            "max_k": 99,
            "c": 0,
            "targetShape": "2",
            "distractorShape": "5",
            "shapeSize": 20,
            "theta_min": 0,
            "theta_max": 360,
            "targetColour": "#00FF00",
            "distractorColour": "",
            "quadrantOrder": [1, 2, 3, 4],
            "debug": False,
            "present": True,
            "conjunctive": False,
        },
        "2Among5NoColourDn": {
            "num_images": args.number if args.number is not None else 1000,
            "min_k": 0,
            "max_k": args.distractors if args.distractors is not None else 99,
            "c": 0,
            "targetShape": "2",
            "distractorShape": "5",
            "shapeSize": 20,
            "theta_min": 0,
            "theta_max": 360,
            "targetColour": "#00FF00",
            "distractorColour": "",
            "quadrantOrder": [1, 2, 3, 4],
            "debug": False,
            "present": False,
            "conjunctive": False,
        },
        "VerticalGradient": {
            "num_images": args.number if args.number is not None else 1000,
            "min_k": 0,
            "max_k": 49,
            "c": 0,
            "targetShape": "circle_gradient_top_bottom",
            "distractorShape": "circle_gradient_bottom_top",
            "shapeSize": 30,
            "theta_min": 0,
            "theta_max": 0,
            "targetColour": "#000000",
            "distractorColour": "",
            "quadrantOrder": [1, 2, 3, 4],
            "debug": False,
            "present": False,
            "conjunctive": False,
        },
        "HorizontalGradient": {
            "num_images": args.number if args.number is not None else 1000,
            "min_k": 0,
            "max_k": 49,
            "c": 0,
            "targetShape": "circle_gradient_top_bottom",
            "distractorShape": "circle_gradient_bottom_top",
            "shapeSize": 30,
            "theta_min": 90,
            "theta_max": 90,
            "targetColour": "#000000",
            "distractorColour": "",
            "quadrantOrder": [1, 2, 3, 4],
            "debug": False,
            "present": False,
            "conjunctive": False,
        },
        # New TandL preset:
        "TandL": {
            "num_images": args.number if args.number is not None else 1000,
            "min_k": 0,
            "max_k": 10,
            "c": 1,
            "targetShape": "T",
            "distractorShape": "L",
            "shapeSize": 20,
            "theta_min": 0,
            "theta_max": 360,
            "targetColour": "#00FF00",
            "distractorColour": "#0000FF",
            "quadrantOrder": [1, 2, 3, 4],
            "debug": False,
            "present": False,
            "conjunctive": False,
        }
    }
    
    # If a preset is specified and exists, start with that configuration.
    # Otherwise, use a default configuration based on the flags.
    if args.preset and args.preset in presets:
            config = presets[args.preset].copy()
    else:
        config = {
            "num_images": args.number if args.number is not None else 1000,
            "min_k": 0,
            "max_k": args.distractors if args.distractors is not None else 99,
            "c": args.colour if args.colour is not None else 0,
            "targetShape": args.target if args.target is not None else "2",
            "distractorShape": args.distractor if args.distractor is not None else "5",
            "shapeSize": args.size if args.size is not None else 20,
            "theta_min": 0,
            "theta_max": args.rotation if args.rotation is not None else 360,
            "targetColour": args.targetColour if args.targetColour is not None else "#00FF00",
            "distractorColour": args.distractorColour if args.distractorColour is not None else "#0000FF",
            "quadrantOrder": quadrantOrder,
            "debug": args.debug,
            "present": args.present,
            "conjunctive": args.conjunctive,
            "colourMode": args.colourMode if args.colourMode is not None else "explicit"
        }

    # Override preset values with any command-line flags (if not None):
    overrides = {}
    if args.number is not None:
        overrides["num_images"] = args.number
    if args.distractors is not None:
        overrides["max_k"] = args.distractors
    if args.colour is not None:
        overrides["c"] = args.colour
    if args.target is not None:
        overrides["targetShape"] = args.target
    if args.distractor is not None:
        overrides["distractorShape"] = args.distractor
    if args.size is not None:
        overrides["shapeSize"] = args.size
    if args.rotation is not None:
        overrides["theta_max"] = args.rotation
    if args.targetColour is not None:
        overrides["targetColour"] = args.targetColour
    if args.distractorColour is not None:
        overrides["distractorColour"] = args.distractorColour
    if args.debug is not None:
        overrides["debug"] = args.debug
    if args.present:
        overrides["present"] = args.present
    if args.conjunctive:
        overrides["conjunctive"] = args.conjunctive

    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    # Add grid configuration:
    config["grid_cols"] = grid_cols
    config["grid_rows"] = grid_rows
    config["quadrantOrder"] = quadrantOrder

    # Pass through the new colour mode parameters:
    config["colourMode"] = args.colourMode
    config["colourList"] = [c.strip() for c in args.colourList.split(",")] if args.colourList else None

    # Set the output directory based on finetuning flag:
    if args.finetuning:
        config["dir"] =  args.filename
    else:
        config["dir"] =  args.filename


    print(config)
    # Finally, call generate_images with the complete configuration:
    generate_images(**config)