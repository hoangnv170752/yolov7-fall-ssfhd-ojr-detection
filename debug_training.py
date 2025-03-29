#!/usr/bin/env python
"""
Debug script for training process.

This script focuses on debugging the training loop to identify why
metrics are all zeros despite videos loading correctly.
"""

import os
import torch
import yaml
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG level for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_debug.log"),
        logging.StreamHandler()
    ]
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Debug training process')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU ID to use')
    
    return parser.parse_args()

def debug_training():
    """Debug the training process."""
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    print(f"Using device: {device}")
    
    # Import necessary modules
    from models.sshfd import SSHFD
    from utils.data_utils import create_dataloaders
    from utils.metrics import compute_accuracy, compute_precision_recall_f1, compute_auc
    
    # Create dataloaders with small batch size and single worker
    config['training']['num_workers'] = 0  # Disable multiprocessing
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # Print dataloader info
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")
    
    # Create model
    model = SSHFD(
        hidden_dim=config['model']['hidden_dim'],
        num_keypoints=config['model']['num_keypoints'],
        temporal_window=config['model']['temporal_window']
    ).to(device)
    
    # Define loss functions
    keypoint_criterion = torch.nn.MSELoss()
    fall_criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, config['training']['fall_class_weight']]).to(device)
    )
    
    # Define optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Debug training loop
    model.train()
    
    # Process only a few batches for debugging
    max_batches = 3
    
    print("\n=== Starting debug training loop ===")
    for batch_idx, data in enumerate(train_loader):
        if batch_idx >= max_batches:
            break
            
        print(f"\nProcessing batch {batch_idx+1}/{max_batches}")
        
        try:
            # Check data format
            if not isinstance(data, tuple) or len(data) != 2:
                print(f"  ERROR: Invalid data format: {type(data)}")
                continue
                
            frames, labels = data
            
            # Print data shapes
            print(f"  Frames shape: {frames.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Labels: {labels}")
            
            # Check for NaN or invalid values
            if torch.isnan(frames).any():
                print("  ERROR: NaN values in frames")
                continue
                
            if torch.isinf(frames).any():
                print("  ERROR: Inf values in frames")
                continue
                
            # Move data to device
            frames = frames.to(device)
            labels = labels.to(device)
            
            # Forward pass
            print("  Running forward pass...")
            keypoints, confidences, fall_preds = model(frames)
            
            # Print output shapes
            print(f"  Keypoints shape: {keypoints.shape}")
            print(f"  Confidences shape: {confidences.shape}")
            print(f"  Fall predictions shape: {fall_preds.shape}")
            print(f"  Fall predictions: {fall_preds}")
            
            # Calculate losses
            # For keypoint loss, we would need ground truth keypoints
            # Here we use a simplified approach with dummy targets
            keypoint_gt = torch.zeros_like(keypoints).to(device)
            confidence_gt = torch.ones_like(confidences).to(device)
            
            kp_loss = keypoint_criterion(keypoints, keypoint_gt)
            fall_loss = fall_criterion(fall_preds, labels)
            
            # Print losses
            print(f"  Keypoint loss: {kp_loss.item()}")
            print(f"  Fall loss: {fall_loss.item()}")
            
            # Combined loss
            loss = kp_loss * config['training']['keypoint_loss_weight'] + \
                   fall_loss * config['training']['fall_loss_weight']
            
            print(f"  Combined loss: {loss.item()}")
            
            # Check for NaN losses
            if torch.isnan(loss).any():
                print("  ERROR: NaN loss values")
                continue
                
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Check gradients
            print("  Checking gradients...")
            for name, param in model.named_parameters():
                if param.grad is None:
                    print(f"  WARNING: No gradient for {name}")
                elif torch.isnan(param.grad).any():
                    print(f"  ERROR: NaN gradient for {name}")
                elif torch.isinf(param.grad).any():
                    print(f"  ERROR: Inf gradient for {name}")
                elif param.grad.abs().max() == 0:
                    print(f"  WARNING: Zero gradient for {name}")
                else:
                    grad_max = param.grad.abs().max().item()
                    grad_mean = param.grad.abs().mean().item()
                    print(f"  {name}: max={grad_max:.6f}, mean={grad_mean:.6f}")
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Compute metrics
            with torch.no_grad():
                # Compute accuracy
                batch_acc = compute_accuracy(fall_preds, labels)
                print(f"  Accuracy: {batch_acc}")
                
                # Compute precision, recall, F1
                preds = torch.softmax(fall_preds, dim=1).detach().cpu().numpy()
                targets = labels.cpu().numpy()
                
                precision, recall, f1 = compute_precision_recall_f1(
                    preds[:, 1], targets, threshold=0.5
                )
                
                print(f"  Precision: {precision}")
                print(f"  Recall: {recall}")
                print(f"  F1: {f1}")
                
                # Compute AUC
                auc = compute_auc(preds[:, 1], targets)
                print(f"  AUC: {auc}")
        
        except Exception as e:
            print(f"  ERROR: Exception in batch {batch_idx}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n=== Debug training loop completed ===")

if __name__ == "__main__":
    debug_training()
