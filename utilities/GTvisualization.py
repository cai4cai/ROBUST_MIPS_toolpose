import json
import os
from PIL import Image, ImageDraw

# color define
POINT_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # red/green/blue/yellow
BBOX_COLOR = (0, 255, 0)  # green bbox

# Skeleton line define
SKELETON = [[0, 1], [1, 2], [1, 3]]

# line color
SKELETON_COLORS = {
    0: {'link': ('entry', 'hinge'), 'id': 0, 'color': (255, 0, 255)},  # Magenta
    1: {'link': ('hinge', 'tip1'), 'id': 1, 'color': (0, 255, 255)},  # Cyan
    2: {'link': ('hinge', 'tip2'), 'id': 2, 'color': (255, 165, 0)}   # Orange
}

def load_coco_annotations(json_file_path):
    """Load COCO annotations from a JSON file."""
    with open(json_file_path, 'r') as f:
        return json.load(f)

def save_image_with_annotations(image_path, annotations, output_image_path):
    """Draw keypoints, skeleton, and bounding boxes on the image and save it."""
    image = Image.open(image_path).convert('RGB')
    image_with_annotations = image.copy()
    draw = ImageDraw.Draw(image_with_annotations)

    for annotation in annotations:
        keypoints = annotation['keypoints']
        bbox = annotation['bbox']

        # Create a dictionary for keypoints to access them by index
        visible_points = {}

        # Draw keypoints with colors in Red-Green-Blue-Yellow order
        if len(keypoints) >= 3:
            for i in range(0, len(keypoints), 3):
                x, y, visibility = keypoints[i:i + 3]
                if visibility >= 1:  # Draw only if visible
                    radius = 5
                    color = POINT_COLORS[i // 3 % len(POINT_COLORS)]  # Cycle through red, green, blue, yellow
                    draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill=color)
                    visible_points[i // 3] = (x, y)  # save visible points

                # elif visibility == 1:  # Draw occluded points in purple
                #     radius = 3
                #     purple_color = (128, 0, 128)  # RGB value for purple
                #     draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill=purple_color)
                #     visible_points[i // 3] = (x, y)  # save occluded points as well

        # Draw skeleton lines between keypoints
        for i, (start, end) in enumerate(SKELETON):
            if start in visible_points and end in visible_points:
                x_start, y_start = visible_points[start]
                x_end, y_end = visible_points[end]
                color = SKELETON_COLORS[i]['color']
                draw.line([(x_start, y_start), (x_end, y_end)], fill=color, width=3)

        # Draw bounding box with green border
        if len(bbox) == 4 and any(bbox[2:]):
            x1, y1, width, height = map(int, bbox)
            draw.rectangle([x1, y1, x1 + width, y1 + height], outline=BBOX_COLOR, width=3)

    image_with_annotations.save(output_image_path)
    print(f"Saved image with annotations to {output_image_path}")

def process_images(json_file_path, output_folder):
    """Process images and annotations from the COCO JSON file."""
    os.makedirs(output_folder, exist_ok=True)
    
    coco_data = load_coco_annotations(json_file_path)
    images = coco_data['images']
    annotations = coco_data['annotations']

    # Group annotations by image
    annotations_by_image = {}
    for annotation in annotations:
        image_id = annotation['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(annotation)

    # Process each image
    for image in images:
        image_id = image['id']
        image_path = image['file_name']
        file_name = os.path.basename(image_path)
        output_image_path = os.path.join(output_folder, file_name)

        if image_id in annotations_by_image:
            save_image_with_annotations(image_path, annotations_by_image[image_id], output_image_path)
        else:
            print(f"No annotations found for image {image_path}")

def process_all_datasets():
    """Process tra, val, and test datasets."""
    datasets = {
        'tra': 'dataset/cocoformat_train.json',
        'val': 'dataset/cocoformat_val.json',
        'test': 'dataset/cocoformat_test.json'
    }

    output_folders = {
        'tra': 'dataset/check_trainImg',
        'val': '/dataset/check_valImg',
        'test': 'dataset/check_testImg'
    }

    for dataset_type, json_file_path in datasets.items():
        output_folder = output_folders[dataset_type]
        print(f"Processing {dataset_type} dataset...")
        process_images(json_file_path, output_folder)
        print(f"Finished processing {dataset_type} dataset")

if __name__ == "__main__":
    process_all_datasets()
