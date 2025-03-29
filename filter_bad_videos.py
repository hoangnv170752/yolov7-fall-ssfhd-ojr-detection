#!/usr/bin/env python
"""
Filter bad videos script.

This script checks all videos in the dataset lists and removes videos
that have frame loading issues or other problems. It creates new filtered
dataset list files that exclude problematic videos.
"""

import os
import cv2
import argparse
from pathlib import Path
import logging
import yaml
import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("video_filtering.log"),
        logging.StreamHandler()
    ]
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Filter bad videos from dataset')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--min-frames', type=int, default=5, help='Minimum number of frames required')
    parser.add_argument('--check-all-frames', action='store_true', help='Check all frames in videos (slower but more thorough)')
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

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
                    label = int(parts[1]) if len(parts) > 1 else None
                    videos.append((video_path, label))
    
    return videos

def check_video(video_path, data_root, min_frames=5, check_all_frames=False):
    """
    Check if video can be loaded and has enough valid frames.
    
    Args:
        video_path: Path to video relative to data_root
        data_root: Root directory containing videos
        min_frames: Minimum number of frames required
        check_all_frames: Whether to check all frames in the video
        
    Returns:
        tuple: (is_valid, reason)
    """
    full_path = os.path.join(data_root, video_path)
    
    if not os.path.exists(full_path):
        return False, f"File not found: {full_path}"
    
    # Try different video backends
    cap = None
    backends = [
        cv2.CAP_FFMPEG,       # Try FFMPEG first (most reliable)
        1200,                 # Try backend 1200 (seen in logs)
        1900,                 # Try backend 1900 (seen in logs)
        cv2.CAP_AVFOUNDATION, # Try AVFoundation (for macOS)
        cv2.CAP_ANY           # Try any available backend
    ]
    
    for backend in backends:
        try:
            cap = cv2.VideoCapture(str(full_path), backend)
            if cap.isOpened():
                # Check if we can read the first frame
                ret, frame = cap.read()
                if ret:
                    break
                cap.release()
                cap = None
        except Exception:
            if cap is not None:
                cap.release()
                cap = None
    
    # If all backends failed, try without specifying a backend
    if cap is None:
        try:
            cap = cv2.VideoCapture(str(full_path))
            if not cap.isOpened():
                return False, "Failed to open with any backend"
        except Exception as e:
            return False, f"Error opening video: {str(e)}"
    
    # Check video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < min_frames:
        cap.release()
        return False, f"Too few frames: {frame_count} < {min_frames}"
    
    # If requested, check if we can read all frames
    if check_all_frames:
        valid_frames = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Check frames at regular intervals to save time
        step = max(1, frame_count // 10)
        for i in range(0, frame_count, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                valid_frames += 1
        
        cap.release()
        
        if valid_frames < min(10, frame_count // step):
            return False, f"Too few valid frames: {valid_frames}"
    else:
        # Just check a few key frames
        frames_to_check = [0, frame_count // 2, frame_count - 1]
        valid_frames = 0
        
        for frame_idx in frames_to_check:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                valid_frames += 1
        
        cap.release()
        
        if valid_frames < len(frames_to_check):
            return False, f"Failed to read key frames: {valid_frames}/{len(frames_to_check)}"
    
    return True, "OK"

def filter_videos(videos, data_root, min_frames=5, check_all_frames=False):
    """
    Filter out bad videos.
    
    Args:
        videos: List of (video_path, label) tuples
        data_root: Root directory containing videos
        min_frames: Minimum number of frames required
        check_all_frames: Whether to check all frames in the video
        
    Returns:
        tuple: (good_videos, bad_videos)
    """
    good_videos = []
    bad_videos = []
    
    for video_path, label in tqdm.tqdm(videos, desc="Checking videos"):
        is_valid, reason = check_video(video_path, data_root, min_frames, check_all_frames)
        if is_valid:
            good_videos.append((video_path, label))
        else:
            bad_videos.append((video_path, label, reason))
            logging.warning(f"Bad video: {video_path} - {reason}")
    
    return good_videos, bad_videos

def save_filtered_list(videos, output_file):
    """Save filtered video list to file."""
    with open(output_file, 'w') as f:
        for video_path, label in videos:
            if label is not None:
                f.write(f"{video_path},{label}\n")
            else:
                f.write(f"{video_path}\n")
    logging.info(f"Saved {len(videos)} videos to {output_file}")

def save_bad_videos_list(bad_videos, output_file):
    """Save list of bad videos with reasons to file."""
    with open(output_file, 'w') as f:
        f.write("video_path,label,reason\n")
        for video_path, label, reason in bad_videos:
            label_str = str(label) if label is not None else ""
            f.write(f"{video_path},{label_str},{reason}\n")
    logging.info(f"Saved {len(bad_videos)} bad videos to {output_file}")

def main():
    """Main function."""
    args = parse_args()
    config = load_config(args.config)
    data_root = config['data']['root']
    
    # Process train list
    train_list = config['data']['train_list']
    train_videos = load_video_list(train_list)
    logging.info(f"Loaded {len(train_videos)} videos from {train_list}")
    
    good_train_videos, bad_train_videos = filter_videos(
        train_videos, data_root, args.min_frames, args.check_all_frames
    )
    
    # Process val list
    val_list = config['data']['val_list']
    val_videos = load_video_list(val_list)
    logging.info(f"Loaded {len(val_videos)} videos from {val_list}")
    
    good_val_videos, bad_val_videos = filter_videos(
        val_videos, data_root, args.min_frames, args.check_all_frames
    )
    
    # Process test list
    test_list = config['data']['test_list']
    test_videos = load_video_list(test_list)
    logging.info(f"Loaded {len(test_videos)} videos from {test_list}")
    
    good_test_videos, bad_test_videos = filter_videos(
        test_videos, data_root, args.min_frames, args.check_all_frames
    )
    
    # Save filtered lists
    train_filtered = train_list.replace('.txt', '_filtered.txt')
    val_filtered = val_list.replace('.txt', '_filtered.txt')
    test_filtered = test_list.replace('.txt', '_filtered.txt')
    
    save_filtered_list(good_train_videos, train_filtered)
    save_filtered_list(good_val_videos, val_filtered)
    save_filtered_list(good_test_videos, test_filtered)
    
    # Save bad videos lists
    bad_train_list = train_list.replace('.txt', '_bad.csv')
    bad_val_list = val_list.replace('.txt', '_bad.csv')
    bad_test_list = test_list.replace('.txt', '_bad.csv')
    
    save_bad_videos_list(bad_train_videos, bad_train_list)
    save_bad_videos_list(bad_val_videos, bad_val_list)
    save_bad_videos_list(bad_test_videos, bad_test_list)
    
    # Print summary
    total_videos = len(train_videos) + len(val_videos) + len(test_videos)
    total_good = len(good_train_videos) + len(good_val_videos) + len(good_test_videos)
    total_bad = len(bad_train_videos) + len(bad_val_videos) + len(bad_test_videos)
    
    print("\nSummary:")
    print(f"Total videos: {total_videos}")
    print(f"Good videos: {total_good} ({total_good/total_videos*100:.1f}%)")
    print(f"Bad videos: {total_bad} ({total_bad/total_videos*100:.1f}%)")
    print("\nFiltered lists saved to:")
    print(f"  {train_filtered}")
    print(f"  {val_filtered}")
    print(f"  {test_filtered}")
    print("\nBad videos lists saved to:")
    print(f"  {bad_train_list}")
    print(f"  {bad_val_list}")
    print(f"  {bad_test_list}")
    
    # Update config file
    print("\nTo use the filtered lists, update your config.yaml file with:")
    print("data:")
    print(f"  train_list: {train_filtered}")
    print(f"  val_list: {val_filtered}")
    print(f"  test_list: {test_filtered}")

if __name__ == "__main__":
    main()
