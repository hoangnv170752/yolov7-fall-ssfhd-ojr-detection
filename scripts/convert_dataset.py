#!/usr/bin/env python
"""
Convert dataset formats for fall detection.

This script converts various dataset formats to the unified format used in this project.
Supported input formats:
- COCO keypoints
- Custom CSV keypoints
- NTU RGB+D skeleton data
"""

import os
import argparse
import json
import csv
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import cv2
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Convert dataset formats for fall detection')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--input-format', type=str, required=True, 
                       choices=['coco', 'csv', 'ntu', 'custom'],
                       help='Input dataset format')
    parser.add_argument('--input-path', type=str, required=True, help='Path to input dataset')
    parser.add_argument('--output-path', type=str, default='data/processed', help='Output directory')
    parser.add_argument('--video-path', type=str, default=None, help='Path to video files (if needed)')
    
    return parser.parse_args()


def convert_coco_keypoints(input_path, output_path, video_path=None):
    """
    Convert COCO keypoints format to project format.
    
    Args:
        input_path: Path to COCO annotation file (JSON)
        output_path: Output directory
        video_path: Path to video files (optional)
    """
    print("Converting COCO keypoints format...")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load COCO annotations
    with open(input_path, 'r') as f:
        coco_data = json.load(f)
    
    # Process annotations
    keypoints_data = []
    
    # Process images
    image_map = {img['id']: img for img in coco_data['images']}
    
    # Process annotations
    for ann in tqdm(coco_data['annotations']):
        if 'keypoints' not in ann or len(ann['keypoints']) == 0:
            continue
        
        image_id = ann['image_id']
        image_info = image_map[image_id]
        filename = image_info['file_name']
        
        # Get keypoints
        # COCO format: [x1, y1, c1, x2, y2, c2, ...]
        # Project format: [x1, y1, x2, y2, ...]
        kpts = ann['keypoints']
        keypoints = []
        confidences = []
        
        for i in range(0, len(kpts), 3):
            x, y, c = kpts[i:i+3]
            
            # Normalize coordinates
            x_norm = x / image_info['width']
            y_norm = y / image_info['height']
            
            keypoints.extend([x_norm, y_norm])
            confidences.append(c / 2.0)  # COCO uses 0=not labeled, 1=labeled but not visible, 2=labeled and visible
        
        # Add to data
        keypoints_data.append({
            'image_id': image_id,
            'filename': filename,
            'keypoints': keypoints,
            'confidence': confidences,
            'bbox': ann['bbox'],
            'category_id': ann['category_id']
        })
    
    # Write output
    output_file = os.path.join(output_path, 'keypoints.csv')
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'filename', 'keypoints', 'confidence', 'bbox', 'category_id'])
        
        for item in keypoints_data:
            writer.writerow([
                item['image_id'],
                item['filename'],
                json.dumps(item['keypoints']),
                json.dumps(item['confidence']),
                json.dumps(item['bbox']),
                item['category_id']
            ])
    
    print(f"Converted {len(keypoints_data)} keypoint annotations to {output_file}")


def convert_csv_keypoints(input_path, output_path, video_path=None):
    """
    Convert custom CSV keypoints format to project format.
    
    Args:
        input_path: Path to CSV file
        output_path: Output directory
        video_path: Path to video files (optional)
    """
    print("Converting custom CSV keypoints format...")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load CSV data
    df = pd.read_csv(input_path)
    
    # Check required columns
    required_columns = ['frame_id', 'video_id', 'keypoints', 'label']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Column '{col}' not found in CSV file")
            return
    
    # Process data
    processed_data = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Get keypoints
        if isinstance(row['keypoints'], str):
            try:
                keypoints = json.loads(row['keypoints'])
            except json.JSONDecodeError:
                print(f"Error: Failed to parse keypoints for row {_}")
                continue
        else:
            keypoints = row['keypoints']
        
        # Get confidence if available
        if 'confidence' in df.columns and isinstance(row['confidence'], str):
            try:
                confidence = json.loads(row['confidence'])
            except json.JSONDecodeError:
                confidence = [1.0] * (len(keypoints) // 2)  # Default confidence
        else:
            confidence = [1.0] * (len(keypoints) // 2)  # Default confidence
        
        # Get bounding box if available
        if 'bbox' in df.columns and isinstance(row['bbox'], str):
            try:
                bbox = json.loads(row['bbox'])
            except json.JSONDecodeError:
                bbox = [0, 0, 1, 1]  # Default bbox (full frame)
        else:
            bbox = [0, 0, 1, 1]  # Default bbox (full frame)
        
        # Add to processed data
        processed_data.append({
            'frame_id': row['frame_id'],
            'video_id': row['video_id'],
            'keypoints': keypoints,
            'confidence': confidence,
            'bbox': bbox,
            'label': row['label']
        })
    
    # Write output
    output_file = os.path.join(output_path, 'keypoints.csv')
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_id', 'video_id', 'keypoints', 'confidence', 'bbox', 'label'])
        
        for item in processed_data:
            writer.writerow([
                item['frame_id'],
                item['video_id'],
                json.dumps(item['keypoints']),
                json.dumps(item['confidence']),
                json.dumps(item['bbox']),
                item['label']
            ])
    
    print(f"Converted {len(processed_data)} keypoint annotations to {output_file}")


def convert_ntu_skeleton(input_path, output_path, video_path=None):
    """
    Convert NTU RGB+D skeleton data to project format.
    
    Args:
        input_path: Path to NTU RGB+D skeleton data directory
        output_path: Output directory
        video_path: Path to video files (optional)
    """
    print("Converting NTU RGB+D skeleton data...")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Get all skeleton files
    skeleton_files = []
    for root, _, files in os.walk(input_path):
        for file in files:
            if file.endswith('.skeleton'):
                skeleton_files.append(os.path.join(root, file))
    
    # Process skeleton files
    processed_data = []
    
    for skeleton_file in tqdm(skeleton_files):
        # Extract video ID from filename
        video_id = os.path.basename(skeleton_file).split('.')[0]
        
        # Parse skeleton file
        with open(skeleton_file, 'r') as f:
            lines = f.readlines()
        
        line_idx = 0
        frame_count = int(lines[line_idx].strip())
        line_idx += 1
        
        for frame in range(frame_count):
            frame_id = frame
            
            # Skip frame index
            line_idx += 1
            
            body_count = int(lines[line_idx].strip())
            line_idx += 1
            
            for body in range(body_count):
                # Skip body info
                line_idx += 1
                
                joint_count = int(lines[line_idx].strip())
                line_idx += 1
                
                keypoints = []
                confidences = []
                
                for joint in range(joint_count):
                    joint_info = lines[line_idx].strip().split()
                    line_idx += 1
                    
                    x, y, z = float(joint_info[0]), float(joint_info[1]), float(joint_info[2])
                    confidence = float(joint_info[3])
                    
                    # Normalize to 0-1 range (NTU uses 3D coordinates)
                    # This is a simplification - proper normalization would require camera parameters
                    x_norm = (x + 1) / 2
                    y_norm = (y + 1) / 2
                    
                    keypoints.extend([x_norm, y_norm])
                    confidences.append(confidence)
                
                # Determine label based on action class from filename
                # NTU RGB+D filename format: S001C002P003R002A001 (A001 is action 1)
                action_class = int(video_id.split('A')[1][:3])
                
                # Classify falls: actions 41-50 are fall actions in NTU RGB+D
                label = 1 if 41 <= action_class <= 50 else 0
                
                # Add to processed data
                processed_data.append({
                    'frame_id': frame_id,
                    'video_id': video_id,
                    'keypoints': keypoints,
                    'confidence': confidences,
                    'bbox': [0, 0, 1, 1],  # Default bbox (full frame)
                    'label': label
                })
    
    # Write output
    output_file = os.path.join(output_path, 'keypoints.csv')
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_id', 'video_id', 'keypoints', 'confidence', 'bbox', 'label'])
        
        for item in processed_data:
            writer.writerow([
                item['frame_id'],
                item['video_id'],
                json.dumps(item['keypoints']),
                json.dumps(item['confidence']),
                json.dumps(item['bbox']),
                item['label']
            ])
    
    print(f"Converted {len(processed_data)} keypoint annotations to {output_file}")


def convert_custom_format(input_path, output_path, video_path=None):
    """
    Convert custom format to project format.
    
    Args:
        input_path: Path to custom format data
        output_path: Output directory
        video_path: Path to video files (required)
    """
    print("Converting custom format...")
    
    if video_path is None:
        print("Error: video_path is required for custom format conversion")
        return
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Check input format and adapt conversion accordingly
    if os.path.isfile(input_path) and input_path.endswith('.json'):
        # Assume JSON format
        with open(input_path, 'r') as f:
            custom_data = json.load(f)
        
        # Process based on your custom format
        # This is just a placeholder - modify according to your format
        processed_data = []
        
        for item in tqdm(custom_data):
            # Extract data based on your format
            frame_id = item.get('frame', 0)
            video_id = item.get('video', '')
            keypoints = item.get('keypoints', [])
            confidence = item.get('confidence', [1.0] * (len(keypoints) // 2))
            bbox = item.get('bbox', [0, 0, 1, 1])
            label = item.get('label', 0)
            
            processed_data.append({
                'frame_id': frame_id,
                'video_id': video_id,
                'keypoints': keypoints,
                'confidence': confidence,
                'bbox': bbox,
                'label': label
            })
    
    elif os.path.isdir(input_path):
        # Assume directory structure with subdirectories for fall/no-fall
        fall_dir = os.path.join(input_path, 'Fall')
        nofall_dir = os.path.join(input_path, 'NoFall')
        
        if not os.path.exists(fall_dir) or not os.path.exists(nofall_dir):
            print("Error: Expected 'Fall' and 'NoFall' subdirectories")
            return
        
        # Process videos and extract keypoints using OpenPose or similar
        # This is just a placeholder - use actual keypoint extraction method
        processed_data = []
        
        # Process fall videos
        for video_file in tqdm(os.listdir(fall_dir)):
            video_path = os.path.join(fall_dir, video_file)
            if not video_path.endswith(('.mp4', '.avi', '.mov')):
                continue
            
            # Extract keypoints (placeholder)
            keypoints = extract_keypoints_from_video(video_path)
            
            for frame_id, frame_keypoints in enumerate(keypoints):
                processed_data.append({
                    'frame_id': frame_id,
                    'video_id': os.path.basename(video_file).split('.')[0],
                    'keypoints': frame_keypoints['points'],
                    'confidence': frame_keypoints['confidence'],
                    'bbox': frame_keypoints['bbox'],
                    'label': 1  # Fall
                })
        
        # Process no-fall videos
        for video_file in tqdm(os.listdir(nofall_dir)):
            video_path = os.path.join(nofall_dir, video_file)
            if not video_path.endswith(('.mp4', '.avi', '.mov')):
                continue
            
            # Extract keypoints (placeholder)
            keypoints = extract_keypoints_from_video(video_path)
            
            for frame_id, frame_keypoints in enumerate(keypoints):
                processed_data.append({
                    'frame_id': frame_id,
                    'video_id': os.path.basename(video_file).split('.')[0],
                    'keypoints': frame_keypoints['points'],
                    'confidence': frame_keypoints['confidence'],
                    'bbox': frame_keypoints['bbox'],
                    'label': 0  # No Fall
                })
    
    else:
        print(f"Error: Unsupported input format: {input_path}")
        return
    
    # Write output
    output_file = os.path.join(output_path, 'keypoints.csv')
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_id', 'video_id', 'keypoints', 'confidence', 'bbox', 'label'])
        
        for item in processed_data:
            writer.writerow([
                item['frame_id'],
                item['video_id'],
                json.dumps(item['keypoints']),
                json.dumps(item['confidence']),
                json.dumps(item['bbox']),
                item['label']
            ])
    
    print(f"Converted {len(processed_data)} keypoint annotations to {output_file}")


def extract_keypoints_from_video(video_path):
    """
    Extract keypoints from video using OpenPose or similar.
    This is a placeholder - implement actual keypoint extraction.
    
    Args:
        video_path: Path to video file
        
    Returns:
        keypoints: List of keypoints for each frame
    """
    # This is a placeholder
    # In a real implementation, you would use OpenPose, MediaPipe, or a custom keypoint detector
    
    cap = cv2.VideoCapture(video_path)
    keypoints = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Placeholder keypoints (17 joints, COCO format)
        frame_keypoints = {
            'points': [0.5, 0.1, 0.5, 0.2, 0.6, 0.2, 0.7, 0.3, 0.4, 0.2, 0.3, 0.3, 
                      0.5, 0.5, 0.6, 0.7, 0.4, 0.7, 0.5, 0.9, 0.5, 1.0, 0.6, 0.9, 
                      0.6, 1.0, 0.4, 0.9, 0.4, 1.0, 0.5, 0.0, 0.5, 0.0],
            'confidence': [0.9, 0.9, 0.8, 0.7, 0.8, 0.7, 0.9, 0.8, 0.8, 0.7, 0.7, 0.6, 0.6, 0.5, 0.5, 0.9, 0.9],
            'bbox': [0.3, 0.1, 0.7, 0.9]
        }
        
        keypoints.append(frame_keypoints)
    
    cap.release()
    return keypoints


def main():
    """Main function."""
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert dataset based on input format
    if args.input_format == 'coco':
        convert_coco_keypoints(args.input_path, args.output_path, args.video_path)
    elif args.input_format == 'csv':
        convert_csv_keypoints(args.input_path, args.output_path, args.video_path)
    elif args.input_format == 'ntu':
        convert_ntu_skeleton(args.input_path, args.output_path, args.video_path)
    elif args.input_format == 'custom':
        convert_custom_format(args.input_path, args.output_path, args.video_path)
    else:
        print(f"Error: Unsupported input format: {args.input_format}")
    
    print("Dataset conversion completed!")


if __name__ == "__main__":
    main()