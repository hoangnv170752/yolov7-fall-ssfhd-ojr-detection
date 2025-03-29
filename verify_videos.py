#!/usr/bin/env python
"""
Video verification script.

This script checks if all videos in the list files can be loaded correctly.
It also counts the total number of MP4 files in the data directory and compares
with the number of files in the list files.
"""

import os
import cv2
import argparse
from pathlib import Path
import logging
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("video_verification.log"),
        logging.StreamHandler()
    ]
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Verify videos in dataset')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--check-all', action='store_true', help='Check all MP4 files in data directory')
    
    return parser.parse_args()

def load_video_list(list_file):
    """Load video list from file."""
    videos = []
    if not os.path.exists(list_file):
        logging.error(f"List file not found: {list_file}")
        return videos
    
    with open(list_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',')
                if len(parts) >= 1:
                    video_path = parts[0]
                    videos.append(video_path)
    
    return videos

def check_video(video_path, data_root):
    """Check if video can be loaded."""
    full_path = os.path.join(data_root, video_path)
    
    if not os.path.exists(full_path):
        logging.error(f"Video file not found: {full_path}")
        return False
    
    # Try different video backends
    for backend in [cv2.CAP_FFMPEG, cv2.CAP_AVFOUNDATION, None]:
        try:
            if backend is None:
                cap = cv2.VideoCapture(str(full_path))
            else:
                cap = cv2.VideoCapture(str(full_path), backend)
            
            if not cap.isOpened():
                continue
            
            # Read the first frame
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                logging.info(f"Successfully loaded video: {video_path} (backend: {backend})")
                return True
            else:
                logging.warning(f"Failed to read frame from video: {video_path} (backend: {backend})")
        except Exception as e:
            logging.warning(f"Error loading video {video_path} with backend {backend}: {str(e)}")
    
    logging.error(f"Failed to load video with any backend: {video_path}")
    return False

def find_all_mp4_files(data_root):
    """Find all MP4 files in data directory."""
    mp4_files = []
    for root, _, files in os.walk(data_root):
        for file in files:
            if file.lower().endswith('.mp4'):
                # Get path relative to data_root
                rel_path = os.path.relpath(os.path.join(root, file), data_root)
                mp4_files.append(rel_path)
    
    return mp4_files

def main():
    """Main function."""
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    data_root = config['data']['root']
    train_list = config['data']['train_list']
    val_list = config['data']['val_list']
    test_list = config['data']['test_list']
    
    # Load video lists
    train_videos = load_video_list(train_list)
    val_videos = load_video_list(val_list)
    test_videos = load_video_list(test_list)
    
    print(f"Found {len(train_videos)} videos in train list")
    print(f"Found {len(val_videos)} videos in val list")
    print(f"Found {len(test_videos)} videos in test list")
    print(f"Total: {len(train_videos) + len(val_videos) + len(test_videos)} videos in lists")
    
    # Find all MP4 files
    all_mp4_files = find_all_mp4_files(data_root)
    print(f"Found {len(all_mp4_files)} MP4 files in data directory")
    
    # Check for videos in lists but not in directory
    all_list_videos = set(train_videos + val_videos + test_videos)
    all_mp4_set = set(all_mp4_files)
    
    missing_videos = all_list_videos - all_mp4_set
    extra_videos = all_mp4_set - all_list_videos
    
    print(f"Videos in lists but not in directory: {len(missing_videos)}")
    for video in sorted(missing_videos):
        print(f"  - {video}")
    
    print(f"Videos in directory but not in lists: {len(extra_videos)}")
    if len(extra_videos) <= 10:
        for video in sorted(extra_videos):
            print(f"  - {video}")
    else:
        for video in sorted(list(extra_videos)[:10]):
            print(f"  - {video}")
        print(f"  ... and {len(extra_videos) - 10} more")
    
    # Check if videos can be loaded
    if args.check_all:
        videos_to_check = all_list_videos
    else:
        # Just check a few videos from each set
        videos_to_check = []
        for videos in [train_videos[:5], val_videos[:5], test_videos[:5]]:
            videos_to_check.extend(videos)
    
    print(f"\nChecking {len(videos_to_check)} videos...")
    success_count = 0
    for video in videos_to_check:
        if check_video(video, data_root):
            success_count += 1
    
    print(f"\nSuccessfully loaded {success_count} out of {len(videos_to_check)} videos")
    print(f"See video_verification.log for details")

if __name__ == "__main__":
    main()
