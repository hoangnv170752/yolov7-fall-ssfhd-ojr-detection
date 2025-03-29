#!/usr/bin/env python
"""
Create a balanced dataset with both fall and non-fall examples.

This script creates synthetic non-fall examples by taking random frames
from fall videos and labeling them as non-falls.
"""

import os
import cv2
import random
import numpy as np
from pathlib import Path
import logging
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("balanced_dataset.log"),
        logging.StreamHandler()
    ]
)

def create_non_fall_video(src_video_path, dst_video_path, frame_count=30, fps=30):
    """
    Create a non-fall video by extracting random frames from a fall video.
    
    Args:
        src_video_path: Path to source fall video
        dst_video_path: Path to destination non-fall video
        frame_count: Number of frames to extract
        fps: Frames per second for output video
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Try different video backends
        for backend in [cv2.CAP_FFMPEG, cv2.CAP_AVFOUNDATION, None]:
            try:
                if backend is None:
                    cap = cv2.VideoCapture(str(src_video_path))
                else:
                    cap = cv2.VideoCapture(str(src_video_path), backend)
                
                if cap.isOpened():
                    break
            except Exception as e:
                logging.warning(f"Error opening video with backend {backend}: {str(e)}")
        
        if not cap.isOpened():
            logging.error(f"Failed to open video: {src_video_path}")
            return False
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < frame_count:
            logging.warning(f"Video has fewer frames than requested: {total_frames} < {frame_count}")
            frame_count = total_frames
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(dst_video_path), fourcc, fps, (width, height))
        
        # Extract random frames
        frames = []
        frame_indices = sorted(random.sample(range(total_frames), frame_count))
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                logging.warning(f"Failed to read frame {idx} from {src_video_path}")
        
        # Write frames to output video
        for frame in frames:
            out.write(frame)
        
        # Release resources
        cap.release()
        out.release()
        
        return True
    
    except Exception as e:
        logging.error(f"Error creating non-fall video: {str(e)}")
        return False

def create_balanced_dataset(data_root, split_ratios={'train': 0.7, 'val': 0.15, 'test': 0.15}):
    """
    Create a balanced dataset with both fall and non-fall examples.
    
    Args:
        data_root: Root directory containing videos
        split_ratios: Ratios for train/val/test splits
    """
    # Create directories for non-fall videos
    non_fall_dirs = {
        'train': Path(data_root) / 'train_mp4_nonfall',
        'val': Path(data_root) / 'val_mp4_nonfall',
        'test': Path(data_root) / 'test_mp4_nonfall'
    }
    
    for dir_path in non_fall_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Get all fall videos
    fall_videos = []
    for split in ['train', 'val', 'test']:
        split_dir = Path(data_root) / f'{split}_mp4'
        if split_dir.exists():
            for video_path in split_dir.glob('*.mp4'):
                fall_videos.append((split, video_path))
    
    logging.info(f"Found {len(fall_videos)} fall videos")
    
    # Shuffle videos
    random.shuffle(fall_videos)
    
    # Create non-fall videos
    non_fall_videos = []
    for i, (split, video_path) in enumerate(fall_videos):
        # Create non-fall video name
        video_name = video_path.name
        non_fall_name = f"nonfall_{video_name}"
        non_fall_path = non_fall_dirs[split] / non_fall_name
        
        # Create non-fall video
        success = create_non_fall_video(video_path, non_fall_path)
        
        if success:
            non_fall_videos.append((split, non_fall_path))
            logging.info(f"Created non-fall video {i+1}/{len(fall_videos)}: {non_fall_path}")
        else:
            logging.error(f"Failed to create non-fall video for {video_path}")
    
    logging.info(f"Created {len(non_fall_videos)} non-fall videos")
    
    # Create balanced list files
    for split in ['train', 'val', 'test']:
        list_file = Path(data_root) / f'{split}_list_balanced.txt'
        
        with open(list_file, 'w') as f:
            # Add fall videos
            for s, video_path in fall_videos:
                if s == split:
                    rel_path = os.path.relpath(video_path, Path(data_root))
                    f.write(f"{rel_path},1\n")
            
            # Add non-fall videos
            for s, video_path in non_fall_videos:
                if s == split:
                    rel_path = os.path.relpath(video_path, Path(data_root))
                    f.write(f"{rel_path},0\n")
        
        logging.info(f"Created balanced list file: {list_file}")

def main():
    """Main function."""
    data_root = 'data'
    create_balanced_dataset(data_root)

if __name__ == "__main__":
    main()
