#!/usr/bin/env python
"""
Create balanced dataset list files with both fall and non-fall examples.

This script creates balanced dataset list files by combining the existing fall videos
with synthetic non-fall examples based on the data_tuple3.csv file.
"""

import os
import pandas as pd
from pathlib import Path
import logging
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("balanced_lists.log"),
        logging.StreamHandler()
    ]
)

def create_balanced_lists(csv_path, data_root):
    """
    Create balanced dataset list files with both fall and non-fall examples.
    
    Args:
        csv_path: Path to data_tuple3.csv
        data_root: Root directory containing videos and list files
    """
    # Load CSV data
    df = pd.read_csv(csv_path)
    
    # Verify column names
    logging.info(f"CSV columns: {df.columns.tolist()}")
    
    # Filter for non-fall segments
    nonfall_df = df[df['label'] == 0]
    
    logging.info(f"Found {len(nonfall_df)} non-fall segments in CSV")
    
    # Group non-fall segments by chute and camera
    nonfall_by_chute_cam = {}
    for idx, row in nonfall_df.iterrows():
        chute = int(row['chute'])
        cam = int(row['cam'])
        key = (chute, cam)
        if key not in nonfall_by_chute_cam:
            nonfall_by_chute_cam[key] = []
        nonfall_by_chute_cam[key].append((int(row['start']), int(row['end'])))
    
    # Get all fall videos
    fall_videos_by_split = {}
    for split in ['train', 'val', 'test']:
        list_path = Path(data_root) / f"{split}_list_mp4.txt"
        fall_videos_by_split[split] = []
        
        if list_path.exists():
            with open(list_path, 'r') as f:
                for line in f:
                    if line.strip():
                        fall_videos_by_split[split].append(line.strip())
        
        logging.info(f"Found {len(fall_videos_by_split[split])} fall videos in {split} split")
    
    # Create synthetic non-fall entries
    nonfall_videos_by_split = {'train': [], 'val': [], 'test': []}
    
    # Assign non-fall segments to splits with the same ratio as fall videos
    total_fall = sum(len(videos) for videos in fall_videos_by_split.values())
    split_ratios = {
        split: len(fall_videos_by_split[split]) / total_fall
        for split in fall_videos_by_split
    }
    
    logging.info(f"Split ratios: {split_ratios}")
    
    # Create synthetic non-fall entries
    for (chute, cam), segments in nonfall_by_chute_cam.items():
        for start, end in segments:
            # Determine which split to assign this non-fall segment to
            split = random.choices(
                list(split_ratios.keys()),
                weights=list(split_ratios.values()),
                k=1
            )[0]
            
            # Create a synthetic non-fall entry
            video_name = f"train_mp4/fall_chute{chute:02d}_cam{cam}.mp4"
            nonfall_videos_by_split[split].append(f"{video_name},0")
    
    # Create balanced list files
    for split in ['train', 'val', 'test']:
        balanced_list_path = Path(data_root) / f"{split}_list_balanced.txt"
        
        with open(balanced_list_path, 'w') as f:
            # Write fall videos
            for video in fall_videos_by_split[split]:
                f.write(f"{video}\n")
            
            # Write non-fall videos
            for video in nonfall_videos_by_split[split]:
                f.write(f"{video}\n")
        
        logging.info(f"Created balanced list file: {balanced_list_path} with {len(fall_videos_by_split[split])} fall and {len(nonfall_videos_by_split[split])} non-fall videos")

def main():
    """Main function."""
    csv_path = 'custom_datasets/data_tuple3.csv'
    data_root = 'data'
    create_balanced_lists(csv_path, data_root)

if __name__ == "__main__":
    main()
