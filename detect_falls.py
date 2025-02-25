#!/usr/bin/env python
"""
Fall Detection Script.

This script processes videos to detect falls using the trained
SSHFD and OJR models, visualizes the results, and optionally
saves the output video.
"""

import os
import cv2
import torch
import numpy as np
import argparse
import yaml
import time
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms as transforms

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from models.sshfd import SSHFD
from models.ojr import OJR
from models.model_utils import load_yolov7_model
from utils.visual_utils import draw_keypoints, draw_fall_alert


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Detect falls in video using SSHFD and OJR models')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--video', type=str, help='Path to input video file')
    parser.add_argument('--camera', type=int, default=-1, help='Camera device ID (default: -1, no camera)')
    parser.add_argument('--output', type=str, default='outputs/output.mp4', help='Path to output video')
    parser.add_argument('--no-display', action='store_true', help='Disable display window')
    parser.add_argument('--threshold', type=float, default=0.5, help='Fall detection threshold')
    
    return parser.parse_args()


def load_models(config, device):
    """
    Load YOLOv7, SSHFD, and OJR models.
    
    Args:
        config: Configuration dictionary
        device: PyTorch device
        
    Returns:
        yolo_model, sshfd_model, ojr_model: Loaded models
    """
    # Load YOLOv7 model
    yolo_model = load_yolov7_model(config['model']['yolo_weights'])
    yolo_model = yolo_model.to(device).eval()
    
    # Load SSHFD model
    sshfd_model = SSHFD(
        num_keypoints=config['model']['num_keypoints'],
        hidden_dim=config['model']['hidden_dim'],
        temporal_window=config['model']['temporal_window']
    )
    sshfd_model.load_state_dict(torch.load(config['model']['save_path'], map_location=device))
    sshfd_model = sshfd_model.to(device).eval()
    
    # Load OJR model
    ojr_model = OJR(
        input_dim=config['model']['num_keypoints'] * 2,
        hidden_dim=config['model']['ojr_hidden_dim'],
        num_layers=config['model']['ojr_num_layers']
    )
    ojr_model.load_state_dict(torch.load(config['model']['ojr_save_path'], map_location=device))
    ojr_model = ojr_model.to(device).eval()
    
    return yolo_model, sshfd_model, ojr_model


def detect_persons(frame, yolo_model, device, conf_thres=0.25, iou_thres=0.45):
    """
    Detect persons in frame using YOLOv7.
    
    Args:
        frame: Input frame (tensor)
        yolo_model: YOLOv7 model
        device: PyTorch device
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        
    Returns:
        persons: Tensor of person bounding boxes [x1, y1, x2, y2, conf]
    """
    # Ensure frame is on the correct device
    frame = frame.to(device)
    
    # Run inference
    with torch.no_grad():
        detections = yolo_model(frame.unsqueeze(0))
        
    # Process detections (usually handled by non_max_suppression in YOLOv7)
    # Here we use a simplified version assuming detections are already processed
    
    # Filter for person class (class 0)
    persons = []
    for detection in detections[0]:
        if detection[5] == 0:  # Class 0 is person in COCO
            x1, y1, x2, y2 = detection[0:4].tolist()
            conf = detection[4].item()
            persons.append([x1, y1, x2, y2, conf])
    
    return torch.tensor(persons).to(device) if persons else torch.zeros((0, 5)).to(device)


def process_video(args, config):
    """
    Process video to detect falls.
    
    Args:
        args: Command line arguments
        config: Configuration dictionary
    """
    # Set device
    device = torch.device(f"cuda:{config['inference']['gpu_id']}" 
                         if torch.cuda.is_available() and config['inference']['gpu_id'] >= 0 
                         else "cpu")
    print(f"Using device: {device}")
    
    # Load models
    yolo_model, sshfd_model, ojr_model = load_models(config, device)
    
    # Initialize input source
    if args.video:
        cap = cv2.VideoCapture(args.video)
        input_source = args.video
    elif args.camera >= 0:
        cap = cv2.VideoCapture(args.camera)
        input_source = f"Camera {args.camera}"
    else:
        print("Error: No input source specified")
        return
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {input_source}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup output video
    output_dir = os.path.dirname(args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*'mp4v'),
        config['inference']['output_fps'] if 'output_fps' in config['inference'] else fps,
        (width, height)
    )
    
    # Initialize transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize frame buffer for temporal analysis
    frame_buffer = []
    keypoint_buffer = []
    confidence_buffer = []
    bbox_buffer = []
    temporal_window = config['model']['temporal_window']
    
    # Process frames
    pbar = tqdm(total=frame_count) if frame_count > 0 else None
    
    frame_idx = 0
    fall_detected = False
    alert_cooldown = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Update progress bar
        if pbar is not None:
            pbar.update(1)
            
        # Preprocess frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(rgb_frame).to(device)
        
        # Person detection
        person_boxes = detect_persons(
            input_tensor, 
            yolo_model, 
            device, 
            conf_thres=config['model']['conf_thres'],
            iou_thres=config['model']['iou_thres']
        )
        
        if len(person_boxes) > 0:
            # Get the person with highest confidence
            best_person = person_boxes[person_boxes[:, 4].argmax()]
            
            # Scale bounding box to frame size
            orig_h, orig_w = frame.shape[:2]
            model_h, model_w = 224, 224  # Input size for the model
            
            x1 = int(best_person[0].item() * orig_w / model_w)
            y1 = int(best_person[1].item() * orig_h / model_h)
            x2 = int(best_person[2].item() * orig_w / model_w)
            y2 = int(best_person[3].item() * orig_h / model_h)
            
            # Ensure bounding box is within frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(orig_w, x2)
            y2 = min(orig_h, y2)
            
            # Add to bbox buffer
            bbox_buffer.append(torch.tensor([x1, y1, x2, y2]).to(device))
            
            # Keypoint estimation using SSHFD
            with torch.no_grad():
                keypoints, confidences, _ = sshfd_model(input_tensor.unsqueeze(0))
            
            # Add to keypoint and confidence buffers
            keypoint_buffer.append(keypoints.squeeze(0))
            confidence_buffer.append(confidences.squeeze(0))
            
            # Add to frame buffer
            frame_buffer.append(input_tensor)
            
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
                        if fall_prob > args.threshold and alert_cooldown == 0:
                            fall_detected = True
                            alert_cooldown = fps * 3  # 3 seconds cooldown
                        
                # Check for occlusions in the last frame
                occlusion_mask = ojr_model.detect_occlusions(
                    keypoint_buffer[-1],
                    confidence_buffer[-1],
                    threshold=config['training']['occlusion_threshold']
                )
                
                # Apply OJR if occlusions detected
                if occlusion_mask.sum() > 0:
                    # Stack keypoint sequence
                    keypoint_seq = torch.stack(keypoint_buffer, dim=0).unsqueeze(0)
                    
                    # Recover occluded keypoints
                    with torch.no_grad():
                        recovered_keypoints = ojr_model(keypoint_seq, occlusion_mask)
                    
                    # Replace occluded keypoints
                    keypoints = keypoint_buffer[-1].clone()
                    keypoints[occlusion_mask > 0] = recovered_keypoints[occlusion_mask > 0]
                else:
                    keypoints = keypoint_buffer[-1]
                
                # Convert keypoints to image coordinates
                scaled_keypoints = keypoints.clone().cpu().numpy()
                scaled_keypoints[0::2] = scaled_keypoints[0::2] * width
                scaled_keypoints[1::2] = scaled_keypoints[1::2] * height
                
                # Visualize results
                output_frame = frame.copy()
                
                # Draw person bounding box
                cv2.rectangle(
                    output_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0) if not fall_detected else (0, 0, 255),
                    2
                )
                
                # Draw keypoints
                draw_keypoints(output_frame, scaled_keypoints, occlusion_mask.cpu().numpy())
                
                # Draw fall alert if detected
                if fall_detected:
                    draw_fall_alert(output_frame, (width, height))
                    
                # Decrement cooldown
                if alert_cooldown > 0:
                    alert_cooldown -= 1
                    if alert_cooldown == 0:
                        fall_detected = False
                
                # Remove oldest frame from buffers
                frame_buffer.pop(0)
                keypoint_buffer.pop(0)
                confidence_buffer.pop(0)
                bbox_buffer.pop(0)
                
            # Display current frame
            if not args.no_display:
                cv2.imshow('Fall Detection', output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Write output frame
            if config['inference']['save_output']:
                output_writer.write(output_frame)
        else:
            # No person detected, use original frame
            output_frame = frame.copy()
            
            # Add empty placeholders to buffers
            if len(frame_buffer) < temporal_window:
                frame_buffer.append(input_tensor)
                keypoint_buffer.append(torch.zeros(config['model']['num_keypoints'] * 2).to(device))
                confidence_buffer.append(torch.zeros(config['model']['num_keypoints']).to(device))
                bbox_buffer.append(torch.zeros(4).to(device))
                
            # Display empty frame
            if not args.no_display:
                cv2.imshow('Fall Detection', output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            # Write output frame
            if config['inference']['save_output']:
                output_writer.write(output_frame)
                
        frame_idx += 1
    
    # Release resources
    cap.release()
    output_writer.release()
    cv2.destroyAllWindows()
    if pbar is not None:
        pbar.close()
        
    print(f"Processing complete. Output saved to {args.output}")


def main():
    """Main function."""
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    process_video(args, config)


if __name__ == "__main__":
    main()