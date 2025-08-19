import os
import shutil
import glob

def copy_and_rename_files(data_root_list, img_output_folder, json_output_folder, file_types, phase):
    """
    Copy and rename files based on the directory structure from a list of root directories.
    Separate output folders for images and JSONs.
    
    Args:
        data_root_list (list): A list of root directories containing the source files.
        img_output_folder (str): The directory to save the renamed image files.
        json_output_folder (str): The directory to save the renamed JSON files.
        file_types (list): List of file extensions (e.g., ['.png', '.json']).
        phase (str): The current phase (e.g., 'training', 'stage1', 'stage2', 'stage3').
    """
    os.makedirs(img_output_folder, exist_ok=True)
    os.makedirs(json_output_folder, exist_ok=True)

    for data_root in data_root_list:
        folder_names = os.listdir(data_root)

        for folder_name in folder_names:
            folder_path = os.path.join(data_root, folder_name)
            
            for file_type in file_types:
                search_pattern = f"**/raw{file_type}"
                file_paths = glob.glob(os.path.join(folder_path, search_pattern), recursive=True)

                for file_path in file_paths:
                    parent_folder_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
                    parent_folder_name_1 = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
                    parent_folder_name_2 = os.path.basename(os.path.dirname(file_path))

                    # Apply different renaming rules based on the phase
                    if phase == 'training':
                        new_file_name = f"Training_{parent_folder_name}_{parent_folder_name_1}_{parent_folder_name_2}{file_type}"
                    elif phase == 'stage1':
                        new_file_name = f"Val_{parent_folder_name}_{parent_folder_name_1}_{parent_folder_name_2}{file_type}"
                    elif phase == 'stage2':
                        new_file_name = f"Testing_stage2_{parent_folder_name}_{parent_folder_name_1}_{parent_folder_name_2}{file_type}"
                    elif phase == 'stage3':
                        new_file_name = f"Testing_stage3_{parent_folder_name}_{parent_folder_name_1}_{parent_folder_name_2}{file_type}"

                    # Separate output folder for images and JSONs
                    if file_type == '.png':
                        new_file_path = os.path.join(img_output_folder, new_file_name)
                    elif file_type == '.json':
                        new_file_path = os.path.join(json_output_folder, new_file_name)
                    
                    # Copy the file to the new location with the new name
                    shutil.copy(file_path, new_file_path)
                    print(f"Copied: {file_path} to {new_file_path}")

# List of directories to process
data_root_list = [
    'dataset/RobustMIPS/Training/Proctocolectomy',
    'dataset/RobustMIPS/Training/Rectal resection',
    'dataset/RobustMIPS/Testing/Stage_1/Proctocolectomy',
    'dataset/RobustMIPS/Testing/Stage_1/Rectal resection',
    'dataset/RobustMIPS/Testing/Stage_2/Proctocolectomy',
    'dataset/RobustMIPS/Testing/Stage_2/Rectal resection',
    'dataset/RobustMIPS/Testing/Stage_3/Sigmoid'
]

# Usage for Training (images and JSONs)
img_output_folder_training = 'dataset/training/img'
json_output_folder_training = 'dataset/training/json'
copy_and_rename_files(data_root_list[:2], img_output_folder_training, json_output_folder_training, ['.png', '.json'], 'training')

# Example usage for Stage 1 (Testing images and JSONs)
img_output_folder_stage1 = 'dataset/val/img'
json_output_folder_stage1 = 'dataset/val/json'
copy_and_rename_files(data_root_list[2:4], img_output_folder_stage1, json_output_folder_stage1, ['.png', '.json'], 'stage1')

# Example usage for Stage 2 (Testing images and JSONs)
img_output_folder_stage2 = 'dataset/testing/img'
json_output_folder_stage2 = 'dataset/testing/json'
copy_and_rename_files(data_root_list[4:6], img_output_folder_stage2, json_output_folder_stage2, ['.png', '.json'], 'stage2')

# Example usage for Stage 3 (Sigmoid images and JSONs)
img_output_folder_stage3 = 'dataset/testing/img'
json_output_folder_stage3 = 'dataset/testing/json'
copy_and_rename_files([data_root_list[6]], img_output_folder_stage3, json_output_folder_stage3, ['.png', '.json'], 'stage3')
