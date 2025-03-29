#!/usr/bin/env python
"""
Create a balanced dataset with both fall and non-fall examples based on data_tuple3.csv.

This script extracts non-fall segments from videos according to the timestamps
in data_tuple3.csv and organizes them into train, validation, and test folders.
"""

import os
import cv2
import numpy as np
import pandas as pd
import random
from pathlib import Path
import logging
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("nonfall_dataset.log"),
        logging.StreamHandler()
    ]
)

def extract_video_segment(video_path, start_frame, end_frame, output_path, fps=30):
    """
    Extract a segment from a video and save it to a new file.
    
    Args:
        video_path: Path to source video
        start_frame: Starting frame number
        end_frame: Ending frame number
        output_path: Path to save the output video
        fps: Frames per second for output video
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Try different video backends
        for backend in [cv2.CAP_FFMPEG, cv2.CAP_AVFOUNDATION, None]:
            try:
                if backend is None:
                    cap = cv2.VideoCapture(str(video_path))
                else:
                    cap = cv2.VideoCapture(str(video_path), backend)
                
                if cap.isOpened():
                    break
            except Exception as e:
                logging.warning(f"Error opening video with backend {backend}: {str(e)}")
        
        if not cap.isOpened():
            logging.error(f"Failed to open video: {video_path}")
            return False
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Set position to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Extract frames
        frame_count = 0
        while frame_count < (end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"Failed to read frame {start_frame + frame_count} from {video_path}")
                break
            
            out.write(frame)
            frame_count += 1
        
        # Release resources
        cap.release()
        out.release()
        
        if frame_count > 0:
            return True
        else:
            logging.error(f"No frames extracted from {video_path}")
            return False
    
    except Exception as e:
        logging.error(f"Error extracting video segment: {str(e)}")
        return False

def find_video_file(chute_num, cam_num, data_root):
    """
    Find the video file for a given chute and camera number.
    
    Args:
        chute_num: Chute number
        cam_num: Camera number
        data_root: Root directory to search in
    
    Returns:
        Path to video file or None if not found
    """
    # Format the filename pattern
    pattern = f"fall_chute{int(chute_num):02d}_cam{int(cam_num)}.mp4"
    
    # Search in train, val, and test directories
    for subdir in ['train_mp4', 'val_mp4', 'test_mp4']:
        video_path = Path(data_root) / subdir / pattern
        if video_path.exists():
            return video_path, subdir
    
    return None, None

def create_nonfall_dataset(csv_path, data_root, output_root):
    """
    Create a dataset of non-fall videos based on data_tuple3.csv.
    
    Args:
        csv_path: Path to data_tuple3.csv
        data_root: Root directory containing original videos
        output_root: Root directory to save non-fall videos
    """
    # Create output directories
    for split in ['train_mp4_nonfall', 'val_mp4_nonfall', 'test_mp4_nonfall']:
        os.makedirs(Path(output_root) / split, exist_ok=True)
    
    # Load CSV data
    df = pd.read_csv(csv_path)
    
    # Verify column names
    logging.info(f"CSV columns: {df.columns.tolist()}")
    
    # Filter for non-fall segments
    nonfall_df = df[df['label'] == 0]
    
    logging.info(f"Found {len(nonfall_df)} non-fall segments in CSV")
    
    # Process each non-fall segment
    success_count = 0
    for idx, row in nonfall_df.iterrows():
        chute_num = row['chute']
        cam_num = row['cam']
        start_frame = int(row['start'])
        end_frame = int(row['end'])
        
        # Find the video file
        video_path, original_split = find_video_file(chute_num, cam_num, data_root)
        
        if video_path is None:
            logging.warning(f"Video not found for chute {chute_num}, camera {cam_num}")
            continue
        
        # Determine output split (keep same as original)
        if original_split == 'train_mp4':
            output_split = 'train_mp4_nonfall'
        elif original_split == 'val_mp4':
            output_split = 'val_mp4_nonfall'
        else:
            output_split = 'test_mp4_nonfall'
        
        # Create output filename
        output_filename = f"nonfall_chute{int(chute_num):02d}_cam{int(cam_num)}_{start_frame}_{end_frame}.mp4"
        output_path = Path(output_root) / output_split / output_filename
        
        # Extract the segment
        success = extract_video_segment(video_path, start_frame, end_frame, output_path)
        
        if success:
            success_count += 1
            logging.info(f"Created non-fall video {success_count}/{len(nonfall_df)}: {output_path}")
        else:
            logging.error(f"Failed to create non-fall video for {video_path}, frames {start_frame}-{end_frame}")
    
    logging.info(f"Successfully created {success_count} non-fall videos")
    
    # Create balanced list files
    for split_type, output_split in [
        ('train', 'train_mp4_nonfall'), 
        ('val', 'val_mp4_nonfall'), 
        ('test', 'test_mp4_nonfall')
    ]:
        # Get original fall videos
        fall_list_path = Path(data_root) / f"{split_type}_list_mp4.txt"
        fall_videos = []
        
        if fall_list_path.exists():
            with open(fall_list_path, 'r') as f:
                for line in f:
                    if line.strip():
                        fall_videos.append(line.strip())
        
        # Get new non-fall videos
        nonfall_dir = Path(output_root) / output_split
        nonfall_videos = []
        
        if nonfall_dir.exists():
            for video_path in nonfall_dir.glob('*.mp4'):
                rel_path = os.path.relpath(video_path, Path(output_root))
                nonfall_videos.append(f"{rel_path},0")
        
        # Create balanced list file
        balanced_list_path = Path(output_root) / f"{split_type}_list_balanced.txt"
        
        with open(balanced_list_path, 'w') as f:
            # Write fall videos
            for video in fall_videos:
                f.write(f"{video}\n")
            
            # Write non-fall videos
            for video in nonfall_videos:
                f.write(f"{video}\n")
        
        logging.info(f"Created balanced list file: {balanced_list_path} with {len(fall_videos)} fall and {len(nonfall_videos)} non-fall videos")

def main():
    """Main function."""
    csv_path = 'custom_datasets/data_tuple3.csv'
    data_root = 'data'
    output_root = 'data'
    create_nonfall_dataset(csv_path, data_root, output_root)

if __name__ == "__main__":
    main()
