#!/usr/bin/env python
"""
Prepare fall detection datasets for training.

This script processes raw fall detection datasets and prepares them for training:
1. Extracts frames from videos
2. Organizes them into train/val/test splits
3. Creates dataset lists for training
"""

import os
import argparse
import cv2
import numpy as np
import pandas as pd
import random
import shutil
from pathlib import Path
from tqdm import tqdm
import yaml
import json


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Prepare fall detection datasets')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--dataset', type=str, required=True, choices=['urfd', 'multicam', 'up-fall', 'le2i', 'custom'],
                       help='Dataset to prepare')
    parser.add_argument('--raw-data', type=str, required=True, help='Path to raw dataset')
    parser.add_argument('--output-dir', type=str, default='data', help='Output directory')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Validation split ratio')
    parser.add_argument('--test-ratio', type=float, default=0.15, help='Test split ratio')
    parser.add_argument('--frames-per-video', type=int, default=16, help='Number of frames to extract per video')
    parser.add_argument('--no-extract', action='store_true', help='Skip frame extraction')
    
    return parser.parse_args()


def extract_frames(video_path, output_dir, num_frames=16, prefix=''):
    """
    Extract frames from a video.
    
    Args:
        video_path: Path to video file
        output_dir: Output directory for frames
        num_frames: Number of frames to extract
        prefix: Frame filename prefix
        
    Returns:
        frame_paths: List of extracted frame paths
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate sampling rate
    if frame_count <= num_frames:
        # If video has fewer frames, extract all
        sample_indices = list(range(frame_count))
    else:
        # Sample frames evenly
        sample_indices = np.linspace(0, frame_count-1, num_frames, dtype=int)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract frames
    frame_paths = []
    
    for i, frame_idx in enumerate(sample_indices):
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Read frame
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Save frame
        frame_path = os.path.join(output_dir, f"{prefix}{i:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
    
    # Release video
    cap.release()
    
    return frame_paths


def split_dataset(videos, val_ratio, test_ratio):
    """
    Split dataset into train/val/test.
    
    Args:
        videos: List of videos
        val_ratio: Validation split ratio
        test_ratio: Test split ratio
        
    Returns:
        train_videos, val_videos, test_videos: Split video lists
    """
    n_videos = len(videos)
    n_val = int(n_videos * val_ratio)
    n_test = int(n_videos * test_ratio)
    n_train = n_videos - n_val - n_test
    
    train_videos = videos[:n_train]
    val_videos = videos[n_train:n_train+n_val]
    test_videos = videos[n_train+n_val:]
    
    return train_videos, val_videos, test_videos


def process_videos(videos, output_dir, frames_per_video, prefix, label, video_list):
    """
    Process videos: extract frames and update video list.
    
    Args:
        videos: List of video paths
        output_dir: Output directory
        frames_per_video: Number of frames to extract per video
        prefix: Frame filename prefix
        label: Video label (0 for no fall, 1 for fall)
        video_list: List to update with processed videos
    """
    for video in tqdm(videos, desc=f"Processing {prefix}videos"):
        # Extract video ID
        video_id = os.path.basename(video).split('.')[0]
        
        # Create output directory for this video
        video_output_dir = os.path.join(output_dir, f"{prefix}{video_id}")
        
        # Extract frames
        extract_frames(video, video_output_dir, frames_per_video)
        
        # Add to video list
        relative_path = os.path.relpath(video_output_dir, os.path.dirname(output_dir))
        video_list.append((relative_path, label))


def save_dataset_lists(output_dir, train_list, val_list, test_list):
    """
    Save dataset lists.
    
    Args:
        output_dir: Output directory
        train_list: List of (video_path, label) for training
        val_list: List of (video_path, label) for validation
        test_list: List of (video_path, label) for testing
    """
    # Save train list
    with open(os.path.join(output_dir, 'train_list.txt'), 'w') as f:
        for video_path, label in train_list:
            f.write(f"{video_path},{label}\n")
    
    # Save val list
    with open(os.path.join(output_dir, 'val_list.txt'), 'w') as f:
        for video_path, label in val_list:
            f.write(f"{video_path},{label}\n")
    
    # Save test list
    with open(os.path.join(output_dir, 'test_list.txt'), 'w') as f:
        for video_path, label in test_list:
            f.write(f"{video_path},{label}\n")


def prepare_urfd_dataset(raw_data_path, output_dir, val_ratio, test_ratio, frames_per_video, no_extract):
    """
    Prepare UR Fall Detection (URFD) dataset.
    
    Args:
        raw_data_path: Path to raw URFD dataset
        output_dir: Output directory
        val_ratio: Validation split ratio
        test_ratio: Test split ratio
        frames_per_video: Number of frames to extract per video
        no_extract: Skip frame extraction if True
    """
    print("Preparing UR Fall Detection dataset...")
    
    # Define paths
    fall_dir = os.path.join(raw_data_path, 'Falls')
    adl_dir = os.path.join(raw_data_path, 'ADL')
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    for directory in [train_dir, val_dir, test_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Get all video files
    fall_videos = [os.path.join(fall_dir, f) for f in os.listdir(fall_dir) if f.endswith('.avi')]
    adl_videos = [os.path.join(adl_dir, f) for f in os.listdir(adl_dir) if f.endswith('.avi')]
    
    # Shuffle videos
    random.shuffle(fall_videos)
    random.shuffle(adl_videos)
    
    # Split into train/val/test
    fall_train, fall_val, fall_test = split_dataset(fall_videos, val_ratio, test_ratio)
    adl_train, adl_val, adl_test = split_dataset(adl_videos, val_ratio, test_ratio)
    
    # Process splits
    train_list = []
    val_list = []
    test_list = []
    
    if not no_extract:
        # Process fall videos
        print("Processing fall videos...")
        process_videos(fall_train, train_dir, frames_per_video, prefix='fall_', label=1, video_list=train_list)
        process_videos(fall_val, val_dir, frames_per_video, prefix='fall_', label=1, video_list=val_list)
        process_videos(fall_test, test_dir, frames_per_video, prefix='fall_', label=1, video_list=test_list)
        
        # Process ADL videos
        print("Processing ADL videos...")
        process_videos(adl_train, train_dir, frames_per_video, prefix='adl_', label=0, video_list=train_list)
        process_videos(adl_val, val_dir, frames_per_video, prefix='adl_', label=0, video_list=val_list)
        process_videos(adl_test, test_dir, frames_per_video, prefix='adl_', label=0, video_list=test_list)
    else:
        # Just create the lists
        for video in fall_train:
            video_id = f"fall_{os.path.basename(video).replace('.avi', '')}"
            train_list.append((os.path.join('train', video_id), 1))
        
        for video in fall_val:
            video_id = f"fall_{os.path.basename(video).replace('.avi', '')}"
            val_list.append((os.path.join('val', video_id), 1))
        
        for video in fall_test:
            video_id = f"fall_{os.path.basename(video).replace('.avi', '')}"
            test_list.append((os.path.join('test', video_id), 1))
        
        for video in adl_train:
            video_id = f"adl_{os.path.basename(video).replace('.avi', '')}"
            train_list.append((os.path.join('train', video_id), 0))
        
        for video in adl_val:
            video_id = f"adl_{os.path.basename(video).replace('.avi', '')}"
            val_list.append((os.path.join('val', video_id), 0))
        
        for video in adl_test:
            video_id = f"adl_{os.path.basename(video).replace('.avi', '')}"
            test_list.append((os.path.join('test', video_id), 0))
    
    # Save dataset lists
    save_dataset_lists(output_dir, train_list, val_list, test_list)
    
    print("URFD dataset preparation completed!")


def prepare_multicam_dataset(raw_data_path, output_dir, val_ratio, test_ratio, frames_per_video, no_extract):
    """
    Prepare Multi-Cam Fall dataset.
    
    Args:
        raw_data_path: Path to raw Multi-Cam dataset
        output_dir: Output directory
        val_ratio: Validation split ratio
        test_ratio: Test split ratio
        frames_per_video: Number of frames to extract per video
        no_extract: Skip frame extraction if True
    """
    print("Preparing Multi-Cam Fall dataset...")
    
    # Define paths
    fall_dir = os.path.join(raw_data_path, 'Falls')
    adl_dir = os.path.join(raw_data_path, 'NotFalls')
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    for directory in [train_dir, val_dir, test_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Get all video files
    fall_videos = []
    for root, _, files in os.walk(fall_dir):
        for file in files:
            if file.endswith('.avi'):
                fall_videos.append(os.path.join(root, file))
    
    adl_videos = []
    for root, _, files in os.walk(adl_dir):
        for file in files:
            if file.endswith('.avi'):
                adl_videos.append(os.path.join(root, file))
    
    # Shuffle videos
    random.shuffle(fall_videos)
    random.shuffle(adl_videos)
    
    # Split into train/val/test
    fall_train, fall_val, fall_test = split_dataset(fall_videos, val_ratio, test_ratio)
    adl_train, adl_val, adl_test = split_dataset(adl_videos, val_ratio, test_ratio)
    
    # Process splits
    train_list = []
    val_list = []
    test_list = []
    
    if not no_extract:
        # Process fall videos
        print("Processing fall videos...")
        process_videos(fall_train, train_dir, frames_per_video, prefix='fall_', label=1, video_list=train_list)
        process_videos(fall_val, val_dir, frames_per_video, prefix='fall_', label=1, video_list=val_list)
        process_videos(fall_test, test_dir, frames_per_video, prefix='fall_', label=1, video_list=test_list)
        
        # Process ADL videos
        print("Processing ADL videos...")
        process_videos(adl_train, train_dir, frames_per_video, prefix='adl_', label=0, video_list=train_list)
        process_videos(adl_val, val_dir, frames_per_video, prefix='adl_', label=0, video_list=val_list)
        process_videos(adl_test, test_dir, frames_per_video, prefix='adl_', label=0, video_list=test_list)
    else:
        # Just create the lists
        # Implementation similar to URFD dataset
        pass
    
    # Save dataset lists
    save_dataset_lists(output_dir, train_list, val_list, test_list)
    
    print("Multi-Cam dataset preparation completed!")


def prepare_le2i_dataset(raw_data_path, output_dir, val_ratio, test_ratio, frames_per_video, no_extract):
    """
    Prepare Le2i Fall dataset.
    
    Args:
        raw_data_path: Path to raw Le2i dataset
        output_dir: Output directory
        val_ratio: Validation split ratio
        test_ratio: Test split ratio
        frames_per_video: Number of frames to extract per video
        no_extract: Skip frame extraction if True
    """
    print("Preparing Le2i Fall dataset...")
    
    # Define folders
    scenarios = ['Coffee_room', 'Home', 'Office', 'Lecture_room']
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    for directory in [train_dir, val_dir, test_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Get all video files
    fall_videos = []
    adl_videos = []
    
    for scenario in scenarios:
        scenario_dir = os.path.join(raw_data_path, scenario)
        if not os.path.exists(scenario_dir):
            continue
        
        # Get all subjects
        for subject in os.listdir(scenario_dir):
            subject_dir = os.path.join(scenario_dir, subject)
            if not os.path.isdir(subject_dir):
                continue
            
            # Get all videos
            for video in os.listdir(subject_dir):
                if not video.endswith('.avi'):
                    continue
                
                video_path = os.path.join(subject_dir, video)
                
                if 'fall' in video.lower():
                    fall_videos.append(video_path)
                else:
                    adl_videos.append(video_path)
    
    # Shuffle videos
    random.shuffle(fall_videos)
    random.shuffle(adl_videos)
    
    # Split into train/val/test
    fall_train, fall_val, fall_test = split_dataset(fall_videos, val_ratio, test_ratio)
    adl_train, adl_val, adl_test = split_dataset(adl_videos, val_ratio, test_ratio)
    
    # Process splits
    train_list = []
    val_list = []
    test_list = []
    
    if not no_extract:
        # Process fall videos
        print("Processing fall videos...")
        process_videos(fall_train, train_dir, frames_per_video, prefix='fall_', label=1, video_list=train_list)
        process_videos(fall_val, val_dir, frames_per_video, prefix='fall_', label=1, video_list=val_list)
        process_videos(fall_test, test_dir, frames_per_video, prefix='fall_', label=1, video_list=test_list)
        
        # Process ADL videos
        print("Processing ADL videos...")
        process_videos(adl_train, train_dir, frames_per_video, prefix='adl_', label=0, video_list=train_list)
        process_videos(adl_val, val_dir, frames_per_video, prefix='adl_', label=0, video_list=val_list)
        process_videos(adl_test, test_dir, frames_per_video, prefix='adl_', label=0, video_list=test_list)
    else:
        # Just create the lists
        # Implementation similar to URFD dataset
        pass
    
    # Save dataset lists
    save_dataset_lists(output_dir, train_list, val_list, test_list)
    
    print("Le2i dataset preparation completed!")


def prepare_custom_dataset(raw_data_path, output_dir, val_ratio, test_ratio, frames_per_video, no_extract):
    """
    Prepare custom dataset.
    
    Args:
        raw_data_path: Path to raw custom dataset
        output_dir: Output directory
        val_ratio: Validation split ratio
        test_ratio: Test split ratio
        frames_per_video: Number of frames to extract per video
        no_extract: Skip frame extraction if True
    """
    print("Preparing custom dataset...")
    
    # Define paths
    fall_dir = os.path.join(raw_data_path, 'Fall')
    nofall_dir = os.path.join(raw_data_path, 'NoFall')
    
    # Check if directories exist
    if not os.path.exists(fall_dir) or not os.path.exists(nofall_dir):
        print(f"Error: Fall or NoFall directory not found in {raw_data_path}")
        print("Please organize your custom dataset as follows:")
        print("custom_dataset/")
        print("├── Fall/")
        print("│   ├── video1.mp4")
        print("│   ├── video2.mp4")
        print("│   └── ...")
        print("└── NoFall/")
        print("    ├── video1.mp4")
        print("    ├── video2.mp4")
        print("    └── ...")
        return
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    for directory in [train_dir, val_dir, test_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Get all video files
    fall_videos = []
    for file in os.listdir(fall_dir):
        if file.endswith(('.mp4', '.avi', '.mov')):
            fall_videos.append(os.path.join(fall_dir, file))
    
    nofall_videos = []
    for file in os.listdir(nofall_dir):
        if file.endswith(('.mp4', '.avi', '.mov')):
            nofall_videos.append(os.path.join(nofall_dir, file))
    
    # Shuffle videos
    random.shuffle(fall_videos)
    random.shuffle(nofall_videos)
    
    # Split into train/val/test
    fall_train, fall_val, fall_test = split_dataset(fall_videos, val_ratio, test_ratio)
    nofall_train, nofall_val, nofall_test = split_dataset(nofall_videos, val_ratio, test_ratio)
    
    # Process splits
    train_list = []
    val_list = []
    test_list = []
    
    if not no_extract:
        # Process fall videos
        print("Processing fall videos...")
        process_videos(fall_train, train_dir, frames_per_video, prefix='fall_', label=1, video_list=train_list)
        process_videos(fall_val, val_dir, frames_per_video, prefix='fall_', label=1, video_list=val_list)
        process_videos(fall_test, test_dir, frames_per_video, prefix='fall_', label=1, video_list=test_list)
        
        # Process no-fall videos
        print("Processing no-fall videos...")
        process_videos(nofall_train, train_dir, frames_per_video, prefix='nofall_', label=0, video_list=train_list)
        process_videos(nofall_val, val_dir, frames_per_video, prefix='nofall_', label=0, video_list=val_list)
        process_videos(nofall_test, test_dir, frames_per_video, prefix='nofall_', label=0, video_list=test_list)
    else:
        # Just create the lists
        # Implementation similar to URFD dataset
        pass
    
    # Save dataset lists
    save_dataset_lists(output_dir, train_list, val_list, test_list)
    
    print("Custom dataset preparation completed!")


def main():
    """Main function."""
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Select dataset preparation function
    if args.dataset == 'urfd':
        prepare_urfd_dataset(args.raw_data, args.output_dir, args.val_ratio, args.test_ratio, 
                           args.frames_per_video, args.no_extract)
    elif args.dataset == 'multicam':
        prepare_multicam_dataset(args.raw_data, args.output_dir, args.val_ratio, args.test_ratio, 
                                args.frames_per_video, args.no_extract)
    elif args.dataset == 'up-fall':
        prepare_upfall_dataset(args.raw_data, args.output_dir, args.val_ratio, args.test_ratio, 
                             args.frames_per_video, args.no_extract)
    elif args.dataset == 'le2i':
        prepare_le2i_dataset(args.raw_data, args.output_dir, args.val_ratio, args.test_ratio, 
                           args.frames_per_video, args.no_extract)
    elif args.dataset == 'custom':
        prepare_custom_dataset(args.raw_data, args.output_dir, args.val_ratio, args.test_ratio, 
                             args.frames_per_video, args.no_extract)
    
    # Update config file with dataset paths
    config['data']['train_list'] = os.path.join(args.output_dir, 'train_list.txt')
    config['data']['val_list'] = os.path.join(args.output_dir, 'val_list.txt')
    config['data']['test_list'] = os.path.join(args.output_dir, 'test_list.txt')
    config['data']['root'] = args.output_dir
    
    # Save updated config
    with open(args.config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Dataset preparation completed. Config file updated: {args.config}")


if __name__ == "__main__":
    main()