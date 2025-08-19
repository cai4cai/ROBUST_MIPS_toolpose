import json
import os
import glob
from PIL import Image
import numpy as np

# Global variables to keep track of image and annotation IDs
IMG_ID = 0
ANN_ID = 0

def reset_global_variables():
    global IMG_ID, ANN_ID
    IMG_ID = 0
    ANN_ID = 0

# Utility functions
def is_keypoint_in_image(keypoint, width, height):
    x, y = keypoint
    return 0 <= x <= width and 0 <= y <= height

def calculate_bbox(x_coords, y_coords, image_width, image_height):
    if not x_coords or not y_coords:
        return [0, 0, 0, 0], 0

    # corresponding to cocoformat_train/val/test
    min_x = max(0, min(x_coords) - 20)
    min_y = max(0, min(y_coords) - 20)
    max_x = min(image_width, max(x_coords) + 20)
    max_y = min(image_height, max(y_coords) + 20)


    width = max_x - min_x
    height = max_y - min_y
    area = (width**2 + height**2) / 2

    return [min_x, min_y, width, height], area

def process_single_keypoint(tag1, tag2, node, reference_node, node_index, reference_index, image_width, image_height, image_path, log_file_path):
    """
    Process a single keypoint based on tag1 and tag2 and log if specific condition is met.
    """
    if not hasattr(process_single_keypoint, "counter"):
        process_single_keypoint.counter = 0

    if tag1 == 'visible' and tag2 == 'visible':
        return node + [2]  # The node is visible, marked as 2

    elif tag1 == 'visible' and tag2 == 'occluded':
        # Increment counter and log the image path when tag2 is occluded
        process_single_keypoint.counter += 1
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"{image_path}\n")
        return node + [1]  # Mark node as 1 (occluded)

    elif tag1 == 'occluded' and tag2 == 'visible':
        return node + [2]  # The node is visible, marked as 2

    elif tag1 == 'occluded' and tag2 == 'occluded':
        # Special condition based on node and reference_node indices
        if node_index in [0, 1]:
            return node + [0], reference_node + [1]
        elif node_index in [2, 3] and reference_index == 1:
            return node + [1]  # Directly increment node by 1
        else:
            return node + [0], reference_node + [0]

    return [0, 0, 0]  # Default return as invisible

# Process all keypoints
def process_keypoints(nodes, tags, image_width, image_height, image_path, log_file_path):
    """
    Process keypoints in the order of node[0], node[1], node[2], node[3].
    """
    keypoints = []
    iter_keypoints = 4

    # First, check if node[0] and node[1] are within image bounds
    node_0_in_image = nodes[0] is not None and is_keypoint_in_image(nodes[0], image_width, image_height)
    node_1_in_image = nodes[1] is not None and is_keypoint_in_image(nodes[1], image_width, image_height)

    if tags[1] == 'occluded' and tags[0] == 'occluded':
        # Only process node[0] and node[1] if both are within the image
        if node_0_in_image and node_1_in_image:
            result_0, result_1 = process_single_keypoint(tags[1], tags[0], nodes[0], nodes[1], 0, 1, image_width, image_height, image_path, log_file_path)
            keypoints += result_0
            keypoints += result_1
            if result_0[2] == 0:
                iter_keypoints -= 1
            if result_1[2] == 0:
                iter_keypoints -= 1
        else:
            # If not in image, add [0,0,0] for occluded nodes
            keypoints += [0, 0, 0]
            keypoints += [0, 0, 0]
            iter_keypoints -= 2
    else:
        # Process node[0]
        if node_0_in_image:
            result = nodes[0] + [2] if tags[0] == 'visible' else [0, 0, 0]
        else:
            result = [0, 0, 0]
        keypoints += result
        if result[2] == 0:
            iter_keypoints -= 1

        # Process node[1]
        # if node_1_in_image:
        #     result = nodes[1] + [2] if tags[1] == 'visible' else [0, 0, 0]
        if node_1_in_image:
            if tags[1] == 'visible':
                result = nodes[1] + [2]
            elif tags[1] == 'occluded':
                result = nodes[1] + [1]
            else:  # tags[1] == 'missing'
                result = [0, 0, 0]
        else:
            result = [0, 0, 0]
        keypoints += result
        if result[2] == 0:
            iter_keypoints -= 1

    # Process nodes[2] and nodes[3] with the existing logic
    # Process node[2]
    if nodes[2] is not None and is_keypoint_in_image(nodes[2], image_width, image_height):
        result = process_single_keypoint(tags[1], tags[2], nodes[2], nodes[1], 2, 1, image_width, image_height, image_path, log_file_path)
        keypoints += result
        if result == [0, 0, 0]:
            iter_keypoints -= 1
    else:
        keypoints += [0, 0, 0]
        iter_keypoints -= 1

    # Process node[3]
    if nodes[3] is not None and is_keypoint_in_image(nodes[3], image_width, image_height):
        result = process_single_keypoint(tags[1], tags[3], nodes[3], nodes[1], 3, 1, image_width, image_height, image_path, log_file_path)
        keypoints += result
        if result == [0, 0, 0]:
            iter_keypoints -= 1
    else:
        keypoints += [0, 0, 0]
        iter_keypoints -= 1

    return keypoints, iter_keypoints


# Create the bbox dictionary
def create_bbox_dict(image_id, area, bbox, keypoints, num_keypoints):
    global ANN_ID
    bbox_dict = {
        'category_id': 1,
        'iscrowd': 0,
        'image_id': image_id,
        'id': ANN_ID,
        'bbox': bbox,
        'area': area,
        'keypoints': keypoints,
        'num_keypoints': num_keypoints
    }
    ANN_ID += 1
    return bbox_dict

# Process annotations
def process_annotations(Mydata, image_id, image_width, image_height, image_path, log_file_path):
    coco_annotations = []

    if len(Mydata) == 0:
        coco_annotations.append(create_bbox_dict(image_id, 0, [0, 0, 0, 0], [0] * 12, 0))
        return coco_annotations

    for each_ann in Mydata:
        nodes = each_ann.get("nodes", [])
        tags = each_ann.get("tags", [])
        transitions = each_ann.get("transitions", [])

        # Collect coordinates for bbox
        x_coords = [vec[0] for tra in transitions for vec in tra]
        y_coords = [vec[1] for tra in transitions for vec in tra]

        # for keypoint, tag in zip(nodes, tags):
        #     if tag in ['visible', 'occluded'] and is_keypoint_in_image(keypoint, image_width, image_height):
        #         x_coords.append(int(keypoint[0]))
        #         y_coords.append(int(keypoint[1]))
        
        for i, (keypoint, tag) in enumerate(zip(nodes, tags)):
            # Skip node[0] if it is occluded
            if i == 0 and tag == 'occluded':
                continue

            if tag in ['visible', 'occluded'] and is_keypoint_in_image(keypoint, image_width, image_height):
                x_coords.append(int(keypoint[0]))
                y_coords.append(int(keypoint[1]))

        bbox, area = calculate_bbox(x_coords, y_coords, image_width, image_height)
        keypoints, num_keypoints = process_keypoints(nodes, tags, image_width, image_height, image_path, log_file_path)

        coco_annotations.append(create_bbox_dict(image_id, area, bbox, keypoints, num_keypoints))

    return coco_annotations

# Process data
def process_data(image_files, json_files, output_json_path, base_folder, log_file_path):
    global IMG_ID
    coco_data = {
        'categories': [{
            'supercategory': 'SurgicalTool',
            'id': 1,
            'name': 'SurgicalTool',
            'keypoints': ['entry', 'hinge', 'tip1', 'tip2'],
            'skeleton': [[0, 1], [1, 2], [1, 3]]
        }],
        'images': [],
        'annotations': []
    }
    
    for img_file, json_file in zip(image_files, json_files):
        with Image.open(img_file) as img:
            width, height = img.size
            img_dict = {
                'file_name': img_file,
                'height': height,
                'width': width,
                'id': IMG_ID
            }
            coco_data['images'].append(img_dict)

        with open(json_file, 'r') as file:
            Mydata = json.load(file)
            annotations = process_annotations(Mydata, IMG_ID, width, height, img_file, log_file_path)
            coco_data['annotations'] += annotations

        IMG_ID += 1

    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

def main():
    base_folder = 'dataset/'
    log_file_path = 'keypoint_occluded_log.txt'  # Log file for counting specific keypoint cases
    datasets = ['train', 'val', 'test']
    json_folder_names = ['training/json', 'val/json', 'testing/json']
    img_folder_names = ['training/img', 'val/img', 'testing/img']
    
    for dataset, json_folder_name, img_folder_name in zip(datasets, json_folder_names, img_folder_names):
        reset_global_variables()
        
        image_folder = os.path.join(base_folder, img_folder_name)
        json_folder = os.path.join(base_folder, json_folder_name)

        img_files = sorted(glob.glob(os.path.join(image_folder, '*.png')))
        json_files = sorted(glob.glob(os.path.join(json_folder, '*.json')))

        output_json_path = os.path.join(base_folder, f'cocoformat_{dataset}.json')
        process_data(img_files, json_files, output_json_path, base_folder, log_file_path)

if __name__ == "__main__":
    main()
