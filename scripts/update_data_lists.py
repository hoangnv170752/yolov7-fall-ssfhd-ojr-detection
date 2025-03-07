#!/usr/bin/env python
"""
Move organized dataset to the data folder and update train/test/val lists.

This script:
1. Moves the dataset_organized directory to the data folder
2. Creates new train_list.txt, test_list.txt, and val_list.txt files
   with paths to the organized video files
"""

import os
import shutil
import argparse
import random
from pathlib import Path
import json

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Update data lists for organized dataset')
    parser.add_argument('--organized-data', type=str, default='dataset_organized',
                       help='Path to organized dataset folder')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Target data directory')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Ratio of data for training (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Ratio of data for validation (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--test-only', action='store_true',
                       help='Generate lists only without moving data (for testing)')
    
    return parser.parse_args()

def move_data_folder(source_dir, target_dir):
    """
    Move organized dataset to the data folder.
    
    Args:
        source_dir: Path to organized dataset folder
        target_dir: Target data directory
    """
    # Get the name of the organized dataset folder
    source_dirname = os.path.basename(os.path.normpath(source_dir))
    target_path = os.path.join(target_dir, source_dirname)
    
    # Check if target already exists
    if os.path.exists(target_path):
        print(f"Warning: {target_path} already exists.")
        user_input = input("Do you want to replace it? (y/n): ").strip().lower()
        if user_input == 'y':
            shutil.rmtree(target_path)
        else:
            print("Skipping data movement, using existing directory.")
            return target_path
    
    # Move the folder
    try:
        shutil.move(source_dir, target_dir)
        print(f"Moved {source_dir} to {target_path}")
    except Exception as e:
        print(f"Error moving folder: {e}")
        print(f"Will use existing path: {source_dir}")
        return source_dir
    
    return target_path

def find_video_files(data_dir):
    """
    Find all video files in the organized data directory.
    
    Args:
        data_dir: Path to organized data directory
        
    Returns:
        fall_videos: List of paths to fall videos
        nofall_videos: List of paths to non-fall videos
    """
    fall_videos = []
    nofall_videos = []
    
    # Path to Fall and NoFall directories
    fall_dir = os.path.join(data_dir, 'Fall')
    nofall_dir = os.path.join(data_dir, 'NoFall')
    
    # Find all video files in Fall directory
    if os.path.exists(fall_dir):
        for root, _, files in os.walk(fall_dir):
            for file in files:
                if file.endswith(('.avi', '.mp4', '.mov')):
                    # Get the relative path from the data directory
                    rel_path = os.path.relpath(os.path.join(root, file), data_dir)
                    fall_videos.append(rel_path)
    else:
        print(f"Warning: Fall directory not found at {fall_dir}")
    
    # Find all video files in NoFall directory
    if os.path.exists(nofall_dir):
        for root, _, files in os.walk(nofall_dir):
            for file in files:
                if file.endswith(('.avi', '.mp4', '.mov')):
                    # Get the relative path from the data directory
                    rel_path = os.path.relpath(os.path.join(root, file), data_dir)
                    nofall_videos.append(rel_path)
    else:
        print(f"Warning: NoFall directory not found at {nofall_dir}")
    
    return fall_videos, nofall_videos

def create_data_lists(data_dir, fall_videos, nofall_videos, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Create train, test, and validation lists.
    
    Args:
        data_dir: Path to data directory
        fall_videos: List of paths to fall videos
        nofall_videos: List of paths to non-fall videos
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        seed: Random seed
        
    Returns:
        train_list: List of (path, label) pairs for training
        val_list: List of (path, label) pairs for validation
        test_list: List of (path, label) pairs for testing
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Shuffle the video lists
    random.shuffle(fall_videos)
    random.shuffle(nofall_videos)
    
    # Split fall videos
    fall_train_count = int(len(fall_videos) * train_ratio)
    fall_val_count = int(len(fall_videos) * val_ratio)
    
    fall_train = fall_videos[:fall_train_count]
    fall_val = fall_videos[fall_train_count:fall_train_count + fall_val_count]
    fall_test = fall_videos[fall_train_count + fall_val_count:]
    
    # Split non-fall videos
    nofall_train_count = int(len(nofall_videos) * train_ratio)
    nofall_val_count = int(len(nofall_videos) * val_ratio)
    
    nofall_train = nofall_videos[:nofall_train_count]
    nofall_val = nofall_videos[nofall_train_count:nofall_train_count + nofall_val_count]
    nofall_test = nofall_videos[nofall_train_count + nofall_val_count:]
    
    # Create lists with labels (1 for fall, 0 for no fall)
    train_list = [(path, 1) for path in fall_train] + [(path, 0) for path in nofall_train]
    val_list = [(path, 1) for path in fall_val] + [(path, 0) for path in nofall_val]
    test_list = [(path, 1) for path in fall_test] + [(path, 0) for path in nofall_test]
    
    # Shuffle again to mix fall and non-fall videos
    random.shuffle(train_list)
    random.shuffle(val_list)
    random.shuffle(test_list)
    
    # Write the lists to files
    write_list_to_file(os.path.join(data_dir, 'train_list.txt'), train_list)
    write_list_to_file(os.path.join(data_dir, 'val_list.txt'), val_list)
    write_list_to_file(os.path.join(data_dir, 'test_list.txt'), test_list)
    
    print(f"Created data lists in {data_dir}:")
    print(f"  Train: {len(train_list)} videos ({len(fall_train)} fall, {len(nofall_train)} non-fall)")
    print(f"  Validation: {len(val_list)} videos ({len(fall_val)} fall, {len(nofall_val)} non-fall)")
    print(f"  Test: {len(test_list)} videos ({len(fall_test)} fall, {len(nofall_test)} non-fall)")
    
    return train_list, val_list, test_list

def write_list_to_file(filepath, data_list):
    """
    Write a data list to a file.
    
    Args:
        filepath: Path to output file
        data_list: List of (path, label) pairs
    """
    with open(filepath, 'w') as f:
        for path, label in data_list:
            f.write(f"{path},{label}\n")

def main():
    """Main function."""
    args = parse_args()
    
    # Ensure source directory exists
    if not os.path.exists(args.organized_data):
        print(f"Error: Organized dataset directory not found: {args.organized_data}")
        return
    
    # Ensure data directory exists
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Move data folder (or use existing)
    if not args.test_only:
        data_path = move_data_folder(args.organized_data, args.data_dir)
    else:
        data_path = args.organized_data
        print(f"Test mode: Using existing path {data_path} without moving data")
    
    # Find all video files
    fall_videos, nofall_videos = find_video_files(data_path)
    
    print(f"Found {len(fall_videos)} fall videos and {len(nofall_videos)} non-fall videos")
    
    # Create data lists
    create_data_lists(
        args.data_dir,
        fall_videos,
        nofall_videos,
        args.train_ratio,
        args.val_ratio,
        args.seed
    )
    
    # Create a summary file
    summary = {
        'dataset_path': data_path,
        'total_videos': len(fall_videos) + len(nofall_videos),
        'fall_videos': len(fall_videos),
        'nofall_videos': len(nofall_videos),
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'test_ratio': 1 - args.train_ratio - args.val_ratio
    }
    
    with open(os.path.join(args.data_dir, 'dataset_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("Process completed successfully!")

if __name__ == "__main__":
    main()