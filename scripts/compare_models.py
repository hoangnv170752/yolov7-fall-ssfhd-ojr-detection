#!/usr/bin/env python
"""
Compare the performance of custom fall detection solution vs. standard YOLOv7.

This script:
1. Runs both models on the same test videos
2. Measures detection accuracy, speed, and other metrics
3. Generates a comparison table
"""

import os
import sys
import time
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from tabulate import tabulate

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from models.sshfd import SSHFD
from models.ojr import OJR
from models.model_utils import load_yolov7_model, non_max_suppression
from utils.data_utils import create_dataloaders
import torchvision.transforms as transforms


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare fall detection models')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--test-videos', type=str, default='data/test_list.txt', help='Path to test video list')
    parser.add_argument('--output-dir', type=str, default='outputs/comparison', help='Output directory')
    parser.add_argument('--num-videos', type=int, default=10, help='Number of videos to analyze')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    
    return parser.parse_args()


def load_models(config, device):
    """
    Load both YOLOv7 and custom models.
    
    Args:
        config: Configuration dictionary
        device: PyTorch device
        
    Returns:
        yolo_model: Standard YOLOv7 model
        custom_model: Custom fall detection models (SSHFD and OJR)
    """
    print("Loading models...")
    
    # Load standard YOLOv7
    yolo_model = load_yolov7_model(config['model']['yolo_weights'])
    yolo_model = yolo_model.to(device).eval()
    
    # Load custom models
    sshfd_model = SSHFD(
        num_keypoints=config['model']['num_keypoints'],
        hidden_dim=config['model']['hidden_dim'],
        temporal_window=config['model']['temporal_window']
    )
    sshfd_model.load_state_dict(torch.load(config['model']['save_path'], map_location=device))
    sshfd_model = sshfd_model.to(device).eval()
    
    ojr_model = OJR(
        input_dim=config['model']['num_keypoints'] * 2,
        hidden_dim=config['model']['ojr_hidden_dim'],
        num_layers=config['model']['ojr_num_layers']
    )
    ojr_model.load_state_dict(torch.load(config['model']['ojr_save_path'], map_location=device))
    ojr_model = ojr_model.to(device).eval()
    
    return yolo_model, (sshfd_model, ojr_model)

def load_test_videos(test_list_path, num_videos=10):
    """
    Load test video paths and labels.
    
    Args:
        test_list_path: Path to test list file
        num_videos: Number of videos to load
        
    Returns:
        video_paths: List of video paths
        video_labels: List of video labels
    """
    # Check if the test list file exists
    if not os.path.exists(test_list_path):
        print(f"Warning: Test list file not found at {test_list_path}")
        # Try to find it in the data directory
        data_dir = 'data'
        if os.path.exists(os.path.join(data_dir, test_list_path)):
            test_list_path = os.path.join(data_dir, test_list_path)
            print(f"Found test list at: {test_list_path}")
        else:
            print(f"Error: Could not locate test list file. Please check the path.")
            return [], []
    
    print(f"Loading test videos from {test_list_path}...")
    
    video_paths = []
    video_labels = []
    
    try:
        with open(test_list_path, 'r') as f:
            lines = f.readlines()
            
            for line in lines[:num_videos]:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    video_path = parts[0]
                    try:
                        label = int(parts[1])
                    except ValueError:
                        # If the label can't be converted to int, assume it's 0 (no fall)
                        print(f"Warning: Invalid label format in line: {line.strip()}")
                        label = 0
                    
                    video_paths.append(video_path)
                    video_labels.append(label)
    except Exception as e:
        print(f"Error reading test list file: {e}")
        return [], []
    
    if not video_paths:
        print("Warning: No videos found in the test list.")
    else:
        print(f"Loaded {len(video_paths)} test videos")
        
        # Check if the video files exist
        valid_videos = []
        valid_labels = []
        for i, (path, label) in enumerate(zip(video_paths, video_labels)):
            # First try directly
            if os.path.exists(path):
                valid_videos.append(path)
                valid_labels.append(label)
                continue
                
            # Try in data directory
            data_path = os.path.join('data', path)
            if os.path.exists(data_path):
                valid_videos.append(data_path)
                valid_labels.append(label)
                print(f"Found video at: {data_path}")
                continue
                
            # Search in common subdirectories
            for root_dir in ['data', 'data/test', 'test']:
                if os.path.exists(root_dir):
                    for subdir, _, files in os.walk(root_dir):
                        video_name = os.path.basename(path)
                        for file in files:
                            if file == video_name:
                                full_path = os.path.join(subdir, file)
                                valid_videos.append(full_path)
                                valid_labels.append(label)
                                print(f"Found video at: {full_path}")
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
            else:
                print(f"Warning: Could not find video {path}")
        
        if len(valid_videos) < len(video_paths):
            print(f"Warning: Only {len(valid_videos)} out of {len(video_paths)} videos were found.")
        
        return valid_videos, valid_labels
    
    return [], []

def process_video_with_yolov7(video_path, yolo_model, device, conf_thres=0.25, iou_thres=0.45):
    """
    Process video with standard YOLOv7 model.
    
    Args:
        video_path: Path to video
        yolo_model: YOLOv7 model
        device: PyTorch device
        conf_thres: Confidence threshold
        iou_thres: IoU threshold
        
    Returns:
        metrics: Dictionary of metrics
    """
    # Check if the video path exists
    if not os.path.exists(video_path):
        print(f"Error: Video path does not exist: {video_path}")
        # Check if the path might be relative to the data directory
        data_dir = 'data'
        if os.path.exists(os.path.join(data_dir, video_path)):
            video_path = os.path.join(data_dir, video_path)
            print(f"Attempting to use path: {video_path}")
        else:
            return None
    
    # Open video
    try:
        if os.path.isdir(video_path):
            # If it's a directory, assume it contains frame images
            frames = []
            for img_file in sorted(os.listdir(video_path)):
                if img_file.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(video_path, img_file)
                    frames.append(cv2.imread(img_path))
        else:
            # It's a video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video {video_path}")
                return None
            
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return None
    
    # Check if we have frames
    if not frames:
        print(f"Error: No frames could be extracted from {video_path}")
        return None
    
    # Initialize metrics
    num_frames = len(frames)
    person_detections = 0
    processing_time = 0
    
    # Process frames
    for frame in frames:
        # Preprocess frame
        img = cv2.resize(frame, (640, 640))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Inference
        start_time = time.time()
        try:
            with torch.no_grad():
                # Handle the output format from the model
                pred = yolo_model(img)
                
                # Handle different output formats from the model
                if isinstance(pred, list):
                    # If pred is a list of tensors, we need to extract the detection tensor
                    if len(pred) > 0 and isinstance(pred[0], torch.Tensor):
                        pred = pred[0]  # Use the first tensor for detections
                    else:
                        # Create an empty tensor if no detections
                        pred = torch.zeros((1, 0, 6), device=device)
                
                # Apply person detection filter (class 0)
                # Since we're using a placeholder model, we'll simulate person detection
                # In a real implementation, you would use non_max_suppression here
                
                # Apply a simple filtering for person class
                persons = []
                # Check if pred is tensor with the right shape
                if isinstance(pred, torch.Tensor) and pred.dim() > 1:
                    # Check if it has class information (at least 6 columns in last dim)
                    if pred.shape[-1] >= 6:
                        # Filter for class 0 (person)
                        for detection in pred[0]:
                            if detection[5] == 0:  # Class 0 is person
                                persons.append(detection[:5])  # Get box and confidence
                
                # Convert to tensor
                person_dets = torch.tensor(persons).to(device) if persons else torch.zeros((0, 5)).to(device)
                
                # Count person detections
                if len(person_dets) > 0:
                    person_detections += 1
                
        except Exception as e:
            print(f"Error during inference: {e}")
            # Continue with next frame
            continue
            
        processing_time += time.time() - start_time
    
    # Calculate metrics
    avg_fps = num_frames / processing_time if processing_time > 0 else 0
    detection_rate = person_detections / num_frames if num_frames > 0 else 0
    
    return {
        'num_frames': num_frames,
        'processing_time': processing_time,
        'avg_fps': avg_fps,
        'detection_rate': detection_rate,
        'person_detections': person_detections
    }

def process_video_with_custom_model(video_path, custom_models, device, conf_thres=0.25, iou_thres=0.45):
    """
    Process video with custom fall detection model.
    
    Args:
        video_path: Path to video
        custom_models: Tuple of (SSHFD model, OJR model)
        device: PyTorch device
        conf_thres: Confidence threshold
        iou_thres: IoU threshold
        
    Returns:
        metrics: Dictionary of metrics
    """
    sshfd_model, ojr_model = custom_models
    temporal_window = sshfd_model.temporal_window
    
    # Check if the video path exists
    if not os.path.exists(video_path):
        print(f"Error: Video path does not exist: {video_path}")
        # Check if the path might be relative to the data directory
        data_dir = 'data'
        if os.path.exists(os.path.join(data_dir, video_path)):
            video_path = os.path.join(data_dir, video_path)
            print(f"Attempting to use path: {video_path}")
        else:
            return None
    
    # Open video
    try:
        if os.path.isdir(video_path):
            # If it's a directory, assume it contains frame images
            frames = []
            for img_file in sorted(os.listdir(video_path)):
                if img_file.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(video_path, img_file)
                    frames.append(cv2.imread(img_path))
        else:
            # It's a video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video {video_path}")
                return None
            
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return None
    
    # Check if we have frames
    if not frames:
        print(f"Error: No frames could be extracted from {video_path}")
        return None
    
    # Initialize metrics
    num_frames = len(frames)
    person_detections = 0
    fall_detections = 0
    occlusion_recoveries = 0
    processing_time = 0
    
    # Transform for model input
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize buffers for temporal analysis
    frame_buffer = []
    keypoint_buffer = []
    confidence_buffer = []
    
    # Process frames
    for frame in frames:
        # Preprocess frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(rgb_frame).to(device)
        
        # Person detection (using placeholder model since we don't have access to actual YOLO model)
        start_time = time.time()
        
        try:
            with torch.no_grad():
                # Simple person detection simulation
                # In a real implementation, this would use the actual YOLO model
                # For now, we'll directly use the SSHFD model to extract keypoints
                
                keypoints, confidences, _ = sshfd_model(input_tensor.unsqueeze(0))
                
                # Simulate person detection (if keypoints are valid)
                found_person = torch.sum(confidences.squeeze(0)) > 0
                
                if found_person:
                    person_detections += 1
                    
                    # Add to buffers
                    frame_buffer.append(input_tensor)
                    keypoint_buffer.append(keypoints.squeeze(0))
                    confidence_buffer.append(confidences.squeeze(0))
                    
                    # Process when buffer is full
                    if len(frame_buffer) == temporal_window:
                        # Stack frames for temporal analysis
                        temporal_frames = torch.stack(frame_buffer, dim=0).unsqueeze(0)
                        
                        # Process through SSHFD for fall detection
                        with torch.no_grad():
                            _, _, fall_preds = sshfd_model(temporal_frames)
                            
                            # Process fall prediction
                            if fall_preds is not None:
                                fall_prob = torch.softmax(fall_preds, dim=1)[0, 1].item()
                                
                                # Check for fall
                                if fall_prob > 0.5:
                                    fall_detections += 1
                        
                        # Check for occlusions
                        occlusion_mask = ojr_model.detect_occlusions(
                            keypoint_buffer[-1],
                            confidence_buffer[-1],
                            threshold=0.3
                        )
                        
                        # Apply OJR if occlusions detected
                        if occlusion_mask.sum() > 0:
                            occlusion_recoveries += 1
                            
                            # Stack keypoint sequence
                            keypoint_seq = torch.stack(keypoint_buffer, dim=0).unsqueeze(0)
                            
                            # Recover occluded keypoints
                            with torch.no_grad():
                                recovered_keypoints = ojr_model(keypoint_seq, occlusion_mask)
                        
                        # Remove oldest frame from buffers
                        frame_buffer.pop(0)
                        keypoint_buffer.pop(0)
                        confidence_buffer.pop(0)
                else:
                    # If no person detected, still update frame buffer
                    if len(frame_buffer) < temporal_window:
                        frame_buffer.append(input_tensor)
                        keypoint_buffer.append(torch.zeros(sshfd_model.num_keypoints * 2).to(device))
                        confidence_buffer.append(torch.zeros(sshfd_model.num_keypoints).to(device))
        
        except Exception as e:
            print(f"Error during inference: {e}")
            # Continue with next frame
            continue
            
        processing_time += time.time() - start_time
    
    # Calculate metrics
    avg_fps = num_frames / processing_time if processing_time > 0 else 0
    detection_rate = person_detections / num_frames if num_frames > 0 else 0
    
    return {
        'num_frames': num_frames,
        'processing_time': processing_time,
        'avg_fps': avg_fps,
        'detection_rate': detection_rate,
        'person_detections': person_detections,
        'fall_detections': fall_detections,
        'occlusion_recoveries': occlusion_recoveries
    }
def compare_models(video_paths, video_labels, yolo_model, custom_models, device, output_dir):
    """
    Compare YOLOv7 and custom model on test videos.
    
    Args:
        video_paths: List of video paths
        video_labels: List of video labels
        yolo_model: YOLOv7 model
        custom_models: Custom models
        device: PyTorch device
        output_dir: Output directory
        
    Returns:
        comparison_results: DataFrame with comparison results
    """
    results = []
    
    for i, (video_path, label) in enumerate(zip(video_paths, video_labels)):
        print(f"Processing video {i+1}/{len(video_paths)}: {video_path}")
        
        # Process with standard YOLOv7
        print("  Using standard YOLOv7...")
        yolo_metrics = process_video_with_yolov7(video_path, yolo_model, device)
        
        # Process with custom model
        print("  Using custom model...")
        custom_metrics = process_video_with_custom_model(video_path, custom_models, device)
        
        if yolo_metrics is None or custom_metrics is None:
            print(f"  Skipping video {video_path} due to processing errors")
            continue
        
        # Combine results
        result = {
            'video_path': video_path,
            'ground_truth': 'Fall' if label == 1 else 'No Fall',
            'yolo_fps': yolo_metrics['avg_fps'],
            'custom_fps': custom_metrics['avg_fps'],
            'yolo_detection_rate': yolo_metrics['detection_rate'],
            'custom_detection_rate': custom_metrics['detection_rate'],
            'custom_fall_detected': custom_metrics['fall_detections'] > 0,
            'custom_occlusion_recoveries': custom_metrics['occlusion_recoveries'],
            'correct_classification': (label == 1 and custom_metrics['fall_detections'] > 0) or 
                                     (label == 0 and custom_metrics['fall_detections'] == 0)
        }
        
        results.append(result)
    
    # Check if we have any results
    if not results:
        print("No videos were successfully processed. Please check your file paths.")
        # Return an empty dataframe with all expected columns
        empty_df = pd.DataFrame(columns=[
            'video_path', 'ground_truth', 'yolo_fps', 'custom_fps',
            'yolo_detection_rate', 'custom_detection_rate',
            'custom_fall_detected', 'custom_occlusion_recoveries',
            'correct_classification'
        ])
        return empty_df
        
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate summary statistics
    yolo_avg_fps = df['yolo_fps'].mean()
    custom_avg_fps = df['custom_fps'].mean()
    yolo_avg_detection = df['yolo_detection_rate'].mean()
    custom_avg_detection = df['custom_detection_rate'].mean()
    accuracy = df['correct_classification'].mean()
    
    # Add summary row
    summary = pd.DataFrame([{
        'video_path': 'AVERAGE',
        'ground_truth': '-',
        'yolo_fps': yolo_avg_fps,
        'custom_fps': custom_avg_fps,
        'yolo_detection_rate': yolo_avg_detection,
        'custom_detection_rate': custom_avg_detection,
        'custom_fall_detected': '-',
        'custom_occlusion_recoveries': df['custom_occlusion_recoveries'].mean(),
        'correct_classification': accuracy
    }])
    
    df = pd.concat([df, summary])
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'model_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Generate HTML table
    html_path = os.path.join(output_dir, 'model_comparison.html')
    with open(html_path, 'w') as f:
        f.write("""
        <html>
        <head>
            <style>
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:hover { background-color: #f5f5f5; }
                .correct { background-color: #d4edda; }
                .incorrect { background-color: #f8d7da; }
                tr:last-child { font-weight: bold; background-color: #e9ecef; }
            </style>
        </head>
        <body>
            <h1>Model Comparison: Standard YOLOv7 vs. Custom Fall Detection</h1>
        """)
        
        f.write(df.to_html(index=False, classes='table table-striped', border=0))
        
        # Only add visualizations if we have data
        if len(df) > 1:  # If we have more than just the summary row
            # Add some visualizations
            f.write("""
                <h2>Performance Comparison</h2>
                <div style="display: flex; justify-content: space-around;">
                    <div>
                        <h3>Processing Speed (FPS)</h3>
                        <img src="fps_comparison.png" style="max-width: 500px;">
                    </div>
                    <div>
                        <h3>Detection Rate</h3>
                        <img src="detection_comparison.png" style="max-width: 500px;">
                    </div>
                </div>
            """)
        else:
            f.write("""
                <h2>No Data Available</h2>
                <p>No videos were successfully processed. Please check your file paths.</p>
            """)
            
        f.write("""
        </body>
        </html>
        """)
    
    print(f"HTML report saved to {html_path}")
    
    # Only generate visualizations if we have data
    if len(df) > 1:  # If we have more than just the summary row
        # Generate visualization for FPS comparison
        plt.figure(figsize=(10, 6))
        video_names = [os.path.basename(path)[:15] + '...' if len(os.path.basename(path)) > 15 else os.path.basename(path) 
                      for path in df['video_path'][:-1]]  # Exclude summary row
        
        x = np.arange(len(video_names))
        width = 0.35
        
        plt.bar(x - width/2, df['yolo_fps'][:-1], width, label='YOLOv7')
        plt.bar(x + width/2, df['custom_fps'][:-1], width, label='Custom Model')
        
        plt.xlabel('Videos')
        plt.ylabel('Frames Per Second')
        plt.title('Processing Speed Comparison')
        plt.xticks(x, video_names, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'fps_comparison.png'))
        
        # Generate visualization for detection rate
        plt.figure(figsize=(10, 6))
        
        plt.bar(x - width/2, df['yolo_detection_rate'][:-1], width, label='YOLOv7')
        plt.bar(x + width/2, df['custom_detection_rate'][:-1], width, label='Custom Model')
        
        plt.xlabel('Videos')
        plt.ylabel('Detection Rate')
        plt.title('Person Detection Rate Comparison')
        plt.xticks(x, video_names, rotation=45, ha='right')
        plt.legend()
        plt.ylim(0, 1.05)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'detection_comparison.png'))
    
    # Print summary table to console
    if not df.empty:
        print("\nComparison Summary:")
        print(tabulate(df, headers='keys', tablefmt='pretty'))
    else:
        print("\nNo data available for comparison.")
    
    return df

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    print(f"Using device: {device}")
    
    # Load models
    yolo_model, custom_models = load_models(config, device)
    
    # Load test videos
    video_paths, video_labels = load_test_videos(args.test_videos, args.num_videos)
    
    # Compare models
    comparison_results = compare_models(
        video_paths, 
        video_labels, 
        yolo_model, 
        custom_models, 
        device, 
        args.output_dir
    )
    
    print("Comparison completed!")


if __name__ == "__main__":
    main()