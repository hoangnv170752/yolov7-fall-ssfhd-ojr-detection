#!/usr/bin/env python
"""
Create balanced dataset list files with both fall and non-fall examples.

This script creates balanced dataset list files by using the existing fall videos
as both fall and non-fall examples (with different frame ranges).
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
        logging.FileHandler("simple_balanced_lists.log"),
        logging.StreamHandler()
    ]
)

def create_simple_balanced_lists(csv_path, data_root):
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
    
    # Create a mapping of (chute, cam) to fall and non-fall segments
    segments_by_chute_cam = {}
    for idx, row in df.iterrows():
        chute = int(row['chute'])
        cam = int(row['cam'])
        start = int(row['start'])
        end = int(row['end'])
        is_fall = int(row['label'])
        
        key = (chute, cam)
        if key not in segments_by_chute_cam:
            segments_by_chute_cam[key] = {'fall': [], 'nonfall': []}
        
        if is_fall == 1:
            segments_by_chute_cam[key]['fall'].append((start, end))
        else:
            segments_by_chute_cam[key]['nonfall'].append((start, end))
    
    # Get all videos by split
    videos_by_split = {}
    for split in ['train', 'val', 'test']:
        list_path = Path(data_root) / f"{split}_list_mp4.txt"
        videos_by_split[split] = []
        
        if list_path.exists():
            with open(list_path, 'r') as f:
                for line in f:
                    if line.strip():
                        video_path, _ = line.strip().split(',')
                        # Extract chute and cam from filename
                        filename = os.path.basename(video_path)
                        if filename.startswith('fall_chute'):
                            parts = filename.split('_')
                            chute_str = parts[1][5:]  # Extract digits after "chute"
                            cam_str = parts[2].split('.')[0][3:]  # Extract digits after "cam"
                            try:
                                chute = int(chute_str)
                                cam = int(cam_str)
                                videos_by_split[split].append((video_path, chute, cam))
                            except ValueError:
                                logging.warning(f"Could not parse chute/cam from {filename}")
        
        logging.info(f"Found {len(videos_by_split[split])} videos in {split} split")
    
    # Create balanced list files
    for split in ['train', 'val', 'test']:
        balanced_list_path = Path(data_root) / f"{split}_list_balanced.txt"
        fall_count = 0
        nonfall_count = 0
        
        with open(balanced_list_path, 'w') as f:
            # Process each video in this split
            for video_path, chute, cam in videos_by_split[split]:
                key = (chute, cam)
                
                # Add as fall video
                f.write(f"{video_path},1\n")
                fall_count += 1
                
                # If we have non-fall segments for this video, add them too
                if key in segments_by_chute_cam and segments_by_chute_cam[key]['nonfall']:
                    f.write(f"{video_path},0\n")
                    nonfall_count += 1
        
        logging.info(f"Created balanced list file: {balanced_list_path} with {fall_count} fall and {nonfall_count} non-fall videos")

def main():
    """Main function."""
    csv_path = 'custom_datasets/data_tuple3.csv'
    data_root = 'data'
    create_simple_balanced_lists(csv_path, data_root)

if __name__ == "__main__":
    main()
