# import os
# import shutil
# from pathlib import Path

# # Define source and destination directories
# dest_dir = "/Users/akashi/Desktop/yolov7-fall-ssfhd-ojr-detection/custom_datasets"
# source_dir = "/Users/akashi/Downloads/MultiCam_Organized"

"""
Organize MultiCam dataset for fall detection project.

This script:
1. Organizes the MultiCam dataset into Falls/NotFalls structure
2. Creates appropriate train/val/test splits
3. Prepares the data lists for training
"""

import os
import shutil
import random
import argparse
from pathlib import Path
import re

def parse_args():
    parser = argparse.ArgumentParser(description='Organize MultiCam dataset')
    parser.add_argument('--source-dir', type=str, required=True, 
                        help='Path to MultiCam dataset directory')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Output directory for organized data')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Validation split ratio')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                        help='Test split ratio')
    parser.add_argument('--selected-camera', type=int, default=None,
                        help='Use only specific camera angle (1-8), None for all cameras')
    
    return parser.parse_args()

def organize_dataset(source_dir, output_dir, val_ratio, test_ratio, selected_camera=None):
    # Create output directories
    falls_dir = os.path.join(output_dir, "Falls")
    nofalls_dir = os.path.join(output_dir, "NotFalls")
    
    os.makedirs(falls_dir, exist_ok=True)
    os.makedirs(nofalls_dir, exist_ok=True)
    
    all_videos = []
    
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.avi'):
                if selected_camera is not None:
                    cam_pattern = f"cam{selected_camera}.avi"
                    if not file.endswith(cam_pattern):
                        continue
                
                all_videos.append(os.path.join(root, file))
    
    print(f"Found {len(all_videos)} video files")
    
    fall_videos = []
    nofall_videos = []
    
    for video in all_videos:
        # Get base filename for target
        base_name = os.path.basename(video)
        # Get parent folder name
        folder_name = os.path.basename(os.path.dirname(video))
        
        # Create a unique name combining folder and filename
        unique_name = f"{folder_name}_{base_name}"
        
        fall_videos.append((video, unique_name))
    
    print(f"Classified {len(fall_videos)} fall videos and {len(nofall_videos)} non-fall videos")
    
    random.seed(42)  # For reproducibility
    random.shuffle(fall_videos)
    random.shuffle(nofall_videos)
    
    # Calculate split sizes
    fall_count = len(fall_videos)
    fall_val_size = int(fall_count * val_ratio)
    fall_test_size = int(fall_count * test_ratio)
    fall_train_size = fall_count - fall_val_size - fall_test_size
    
    nofall_count = len(nofall_videos)
    nofall_val_size = int(nofall_count * val_ratio)
    nofall_test_size = int(nofall_count * test_ratio)
    nofall_train_size = nofall_count - nofall_val_size - nofall_test_size
    
    # Split falls
    fall_train = fall_videos[:fall_train_size]
    fall_val = fall_videos[fall_train_size:fall_train_size + fall_val_size]
    fall_test = fall_videos[fall_train_size + fall_val_size:]
    
    # Split non-falls
    nofall_train = nofall_videos[:nofall_train_size]
    nofall_val = nofall_videos[nofall_train_size:nofall_train_size + nofall_val_size]
    nofall_test = nofall_videos[nofall_train_size + nofall_val_size:]
    
    # Create train, val, test directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Copy videos to their destinations
    train_list = []
    val_list = []
    test_list = []
    
    # Process falls
    print("Processing fall videos...")
    process_video_set(fall_train, train_dir, "fall", train_list)
    process_video_set(fall_val, val_dir, "fall", val_list)
    process_video_set(fall_test, test_dir, "fall", test_list)
    
    # Process non-falls
    print("Processing non-fall videos...")
    process_video_set(nofall_train, train_dir, "nofall", train_list)
    process_video_set(nofall_val, val_dir, "nofall", val_list)
    process_video_set(nofall_test, test_dir, "nofall", test_list)
    
    # Write list files
    write_list_file(os.path.join(output_dir, 'train_list.txt'), train_list)
    write_list_file(os.path.join(output_dir, 'val_list.txt'), val_list)
    write_list_file(os.path.join(output_dir, 'test_list.txt'), test_list)
    
    print(f"Dataset organization complete! Files in {output_dir}")
    print(f"Training: {len(train_list)} videos")
    print(f"Validation: {len(val_list)} videos")
    print(f"Testing: {len(test_list)} videos")

def process_video_set(video_set, output_dir, prefix, list_output):
    for src_path, unique_name in video_set:
        # Create output path
        dest_path = os.path.join(output_dir, f"{prefix}_{unique_name}")
        dest_dir = os.path.dirname(dest_path)
        
        os.makedirs(dest_dir, exist_ok=True)
        
        # Copy video file
        shutil.copy2(src_path, dest_path)
        
        # Add to list
        rel_path = os.path.relpath(dest_path, os.path.dirname(output_dir))
        label = 1 if prefix == "fall" else 0
        list_output.append((rel_path, label))

def write_list_file(path, items):
    with open(path, 'w') as f:
        for item_path, label in items:
            f.write(f"{item_path},{label}\n")

if __name__ == "__main__":
    args = parse_args()
    organize_dataset(args.source_dir, args.output_dir, args.val_ratio, 
                    args.test_ratio, args.selected_camera)
