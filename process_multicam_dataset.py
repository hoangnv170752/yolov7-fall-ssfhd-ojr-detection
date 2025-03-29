"""
Process MultiCam dataset for fall detection project.

This script:
1. Processes the MultiCam dataset from custom_datasets
2. Organizes it into train/val/test splits in the data folder
3. Generates txt files with file paths and labels
"""

import os
import shutil
import random
import csv
import glob
from pathlib import Path

def process_dataset(custom_datasets_dir, output_dir, val_ratio=0.15, test_ratio=0.15):
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Read data_tuple3.csv to get file paths and labels
    data_csv_path = os.path.join(custom_datasets_dir, 'data_tuple3.csv')
    all_files = []
    
    try:
        with open(data_csv_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                if len(row) >= 2:  # Ensure row has at least path and label
                    file_path = row[0]
                    label = int(row[1])  # Assuming 1 for fall, 0 for no fall
                    all_files.append((file_path, label))
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        
        # Alternative: If CSV reading fails, try to find files directly
        print("Attempting to find files directly from folders...")
        for chute_dir in sorted(glob.glob(os.path.join(custom_datasets_dir, 'chute*'))):
            chute_name = os.path.basename(chute_dir)
            for cam_file in glob.glob(os.path.join(chute_dir, '*.avi')):
                # Determine if it's a fall based on folder name
                # This is a simplification - you may need to adjust based on your dataset
                label = 1  # Assuming all are falls in this dataset
                all_files.append((cam_file, label))
    
    print(f"Found {len(all_files)} files")
    
    # Split into train, val, test
    random.seed(42)  # For reproducibility
    random.shuffle(all_files)
    
    total_count = len(all_files)
    val_size = int(total_count * val_ratio)
    test_size = int(total_count * test_ratio)
    train_size = total_count - val_size - test_size
    
    train_files = all_files[:train_size]
    val_files = all_files[train_size:train_size + val_size]
    test_files = all_files[train_size + val_size:]
    
    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    
    # Process files for each split
    train_list = process_file_set(train_files, train_dir, custom_datasets_dir)
    val_list = process_file_set(val_files, val_dir, custom_datasets_dir)
    test_list = process_file_set(test_files, test_dir, custom_datasets_dir)
    
    # Write list files
    write_list_file(os.path.join(output_dir, 'train_list.txt'), train_list)
    write_list_file(os.path.join(output_dir, 'val_list.txt'), val_list)
    write_list_file(os.path.join(output_dir, 'test_list.txt'), test_list)
    
    print(f"Dataset processing complete! Files in {output_dir}")
    print(f"Training: {len(train_list)} files")
    print(f"Validation: {len(val_list)} files")
    print(f"Testing: {len(test_list)} files")

def process_file_set(file_set, output_dir, source_base_dir):
    list_output = []
    
    for src_path, label in file_set:
        # Handle both absolute and relative paths
        if not os.path.isabs(src_path):
            src_path = os.path.join(source_base_dir, src_path)
        
        if not os.path.exists(src_path):
            print(f"Warning: Source file not found: {src_path}")
            continue
        
        # Create a unique destination filename
        # Extract chute and camera info from path
        path_parts = Path(src_path).parts
        chute_part = next((part for part in path_parts if part.startswith('chute')), 'unknown')
        filename = os.path.basename(src_path)
        
        # Create destination path
        label_prefix = "fall" if label == 1 else "nofall"
        dest_filename = f"{label_prefix}_{chute_part}_{filename}"
        dest_path = os.path.join(output_dir, dest_filename)
        
        # Copy file
        try:
            shutil.copy2(src_path, dest_path)
            
            # Add to list
            rel_path = os.path.relpath(dest_path, os.path.dirname(output_dir))
            list_output.append((rel_path, label))
        except Exception as e:
            print(f"Error copying file {src_path} to {dest_path}: {e}")
    
    return list_output

def write_list_file(path, items):
    with open(path, 'w') as f:
        for item_path, label in items:
            f.write(f"{item_path},{label}\n")

if __name__ == "__main__":
    custom_datasets_dir = "/Users/akashi/Desktop/yolov7-fall-ssfhd-ojr-detection/custom_datasets"
    output_dir = "/Users/akashi/Desktop/yolov7-fall-ssfhd-ojr-detection/data"
    
    process_dataset(custom_datasets_dir, output_dir)
