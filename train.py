#!/usr/bin/env python
"""
Training script for SSHFD and OJR models.

This script trains the SSHFD model for fall detection and the
OJR model for occluded joint recovery.
"""

import os

# Monkey patch torch.save to handle directory creation
import os
import torch

original_save = torch.save

def patched_save(obj, f, *args, **kwargs):
    if isinstance(f, str):
        os.makedirs(os.path.dirname(f), exist_ok=True)
    return original_save(obj, f, *args, **kwargs)

torch.save = patched_save
import torch
import numpy as np
import argparse
import yaml
import time
import logging
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Import project modules
from models.sshfd import SSHFD
from models.ojr import OJR
from models.model_utils import init_weights, load_checkpoint, save_checkpoint
from utils.data_utils import create_dataloaders
from utils.metrics import compute_accuracy, compute_precision_recall_f1, compute_auc
from utils.losses import FocalLoss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_errors.log"),
        logging.StreamHandler()
    ]
)

# Ensure weights directory exists
os.makedirs("weights", exist_ok=True)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train SSHFD and OJR models')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--train-sshfd', action='store_true', help='Train SSHFD model')
    parser.add_argument('--train-ojr', action='store_true', help='Train OJR model')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    
    return parser.parse_args()


def train_sshfd(config, args):
    """
    Train SSHFD model.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    print("=== Training SSHFD Model ===")
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = SSHFD(
        hidden_dim=config['model']['hidden_dim'],
        num_keypoints=config['model']['num_keypoints'],
        temporal_window=config['model']['temporal_window']
    ).to(device)
    
    # Initialize model weights
    init_weights(model)
    
    # Define loss functions
    keypoint_criterion = torch.nn.MSELoss()
    
    # Use Focal Loss for fall classification to handle class imbalance
    class_weights = torch.tensor([1.0, config['training']['fall_class_weight']]).to(device)
    fall_criterion = FocalLoss(
        alpha=1.0,
        gamma=2.0,  # Focusing parameter - higher values give more weight to hard examples
        weight=class_weights
    ).to(device)
    
    print(f"Using Focal Loss for fall classification with gamma=2.0 and class weights={class_weights}")
    
    # Define optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3, 
        verbose=True
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # Create TensorBoard writer
    log_dir = Path("logs/sshfd_training")
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # Training parameters
    num_epochs = config['training']['epochs']
    keypoint_loss_weight = config['training']['keypoint_loss_weight']
    fall_loss_weight = config['training']['fall_loss_weight']
    
    # Track best model
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_keypoint_loss = 0.0
        train_fall_loss = 0.0
        train_preds = []
        train_targets = []
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, (frames, labels) in enumerate(train_bar):
            frames = frames.to(device)
            labels = labels.to(device)
            
            # Forward pass
            model_outputs = model(frames)
            
            # Handle different output formats
            if isinstance(model_outputs, tuple) and len(model_outputs) == 3:
                keypoints, confidences, fall_logits = model_outputs
            elif isinstance(model_outputs, tuple) and len(model_outputs) == 2:
                keypoints, fall_logits = model_outputs
                confidences = torch.ones_like(keypoints[:, :, 0])  # Dummy confidences
            else:
                logging.error(f"Unexpected model output format: {type(model_outputs)}")
                continue
            
            # Compute losses
            kp_loss = keypoint_criterion(keypoints, keypoints)  # Dummy loss for now
            fall_loss = fall_criterion(fall_logits, labels)
            
            # Weighted loss
            loss = keypoint_loss_weight * kp_loss + fall_loss_weight * fall_loss
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_keypoint_loss += kp_loss.item()
            train_fall_loss += fall_loss.item()
            
            # Store predictions and targets for metrics computation
            _, preds = torch.max(fall_logits, 1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
            
            # Update progress bar
            train_bar.set_postfix({
                'loss': loss.item(),
                'kp_loss': kp_loss.item(),
                'fall_loss': fall_loss.item(),
                'acc': (preds == labels).float().mean().item()
            })
        
        # Compute training metrics
        train_loss /= len(train_loader)
        train_keypoint_loss /= len(train_loader)
        train_fall_loss /= len(train_loader)
        
        train_acc = compute_accuracy(train_preds, train_targets)
        train_precision, train_recall, train_f1 = compute_precision_recall_f1(train_preds, train_targets)
        train_auc = compute_auc(train_preds, train_targets)
        
        # Log training metrics to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/train_keypoint', train_keypoint_loss, epoch)
        writer.add_scalar('Loss/train_fall', train_fall_loss, epoch)
        writer.add_scalar('Metrics/train_accuracy', train_acc, epoch)
        writer.add_scalar('Metrics/train_precision', train_precision, epoch)
        writer.add_scalar('Metrics/train_recall', train_recall, epoch)
        writer.add_scalar('Metrics/train_f1', train_f1, epoch)
        writer.add_scalar('Metrics/train_auc', train_auc, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_keypoint_loss = 0.0
        val_fall_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for batch_idx, (frames, labels) in enumerate(val_bar):
                frames = frames.to(device)
                labels = labels.to(device)
                
                # Forward pass
                model_outputs = model(frames)
                
                # Handle different output formats
                if isinstance(model_outputs, tuple) and len(model_outputs) == 3:
                    keypoints, confidences, fall_logits = model_outputs
                elif isinstance(model_outputs, tuple) and len(model_outputs) == 2:
                    keypoints, fall_logits = model_outputs
                    confidences = torch.ones_like(keypoints[:, :, 0])  # Dummy confidences
                else:
                    logging.error(f"Unexpected model output format: {type(model_outputs)}")
                    continue
                
                # Compute losses
                kp_loss = keypoint_criterion(keypoints, keypoints)  # Dummy loss for now
                fall_loss = fall_criterion(fall_logits, labels)
                
                # Weighted loss
                loss = keypoint_loss_weight * kp_loss + fall_loss_weight * fall_loss
                
                # Update metrics
                val_loss += loss.item()
                val_keypoint_loss += kp_loss.item()
                val_fall_loss += fall_loss.item()
                
                # Store predictions and targets for metrics computation
                _, preds = torch.max(fall_logits, 1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
                
                # Update progress bar
                val_bar.set_postfix({
                    'loss': loss.item(),
                    'kp_loss': kp_loss.item(),
                    'fall_loss': fall_loss.item(),
                    'acc': (preds == labels).float().mean().item()
                })
        
        # Compute validation metrics
        val_loss /= len(val_loader)
        val_keypoint_loss /= len(val_loader)
        val_fall_loss /= len(val_loader)
        
        val_acc = compute_accuracy(val_preds, val_targets)
        val_precision, val_recall, val_f1 = compute_precision_recall_f1(val_preds, val_targets)
        val_auc = compute_auc(val_preds, val_targets)
        
        # Log validation metrics to TensorBoard
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Loss/val_keypoint', val_keypoint_loss, epoch)
        writer.add_scalar('Loss/val_fall', val_fall_loss, epoch)
        writer.add_scalar('Metrics/val_accuracy', val_acc, epoch)
        writer.add_scalar('Metrics/val_precision', val_precision, epoch)
        writer.add_scalar('Metrics/val_recall', val_recall, epoch)
        writer.add_scalar('Metrics/val_f1', val_f1, epoch)
        writer.add_scalar('Metrics/val_auc', val_auc, epoch)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f} (KP: {train_keypoint_loss:.4f}, Fall: {train_fall_loss:.4f})")
        print(f"  Train Metrics: Acc={train_acc:.4f}, P={train_precision:.4f}, R={train_recall:.4f}, F1={train_f1:.4f}, AUC={train_auc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} (KP: {val_keypoint_loss:.4f}, Fall: {val_fall_loss:.4f})")
        print(f"  Val Metrics: Acc={val_acc:.4f}, P={val_precision:.4f}, R={val_recall:.4f}, F1={val_f1:.4f}, AUC={val_auc:.4f}")
        
        # Save best model (based on validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, config['model']['save_path'], optimizer, epoch)
            print("  Saved new best model!")
        
        # Also save model if F1 score improves
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_checkpoint(model, config['model']['save_path'].replace('.pt', '_best_f1.pt'), optimizer, epoch)
            print("  Saved new best F1 model!")
    
    # Close TensorBoard writer
    writer.close()
    
    print("SSHFD Training completed!")
    return model


def train_ojr(config, args, sshfd_model=None):
    """
    Train OJR model.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
        sshfd_model: Pre-trained SSHFD model (if None, will be loaded from weights)
    """
    print("=== Training OJR Model ===")
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(config)
    
    # Load SSHFD model if not provided
    if sshfd_model is None:
        sshfd_model = SSHFD(
            num_keypoints=config['model']['num_keypoints'],
            hidden_dim=config['model']['hidden_dim'],
            temporal_window=config['model']['temporal_window']
        )
        
        # Fix for loading the checkpoint correctly
        checkpoint = torch.load(config['model']['save_path'], map_location=device)
        if 'state_dict' in checkpoint:
            sshfd_model.load_state_dict(checkpoint['state_dict'])
        else:
            sshfd_model.load_state_dict(checkpoint)
        
    sshfd_model = sshfd_model.to(device)
    
    # Set SSHFD to eval mode
    sshfd_model.eval()
    
    # Initialize OJR model
    ojr_model = OJR(
        input_dim=config['model']['num_keypoints'] * 2,
        hidden_dim=config['model']['ojr_hidden_dim'],
        num_layers=config['model']['ojr_num_layers']
    )
    
    # Initialize weights
    init_weights(ojr_model, init_type='xavier')
    
    # Move model to device
    ojr_model = ojr_model.to(device)
    
    # Define optimizer and scheduler
    optimizer = torch.optim.Adam(
        ojr_model.parameters(),
        lr=float(config['training']['ojr_lr']),
        weight_decay=float(config['training']['weight_decay'])
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Loss function
    criterion = torch.nn.MSELoss()
    
    # Initialize training variables
    start_epoch = 0
    best_val_loss = float('inf')
    num_epochs = config['training']['ojr_epochs']
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint_path = config['model']['ojr_save_path']
        if os.path.exists(checkpoint_path):
            start_epoch, best_val_loss = load_checkpoint(ojr_model, optimizer, checkpoint_path)
            print(f"Resuming from epoch {start_epoch} with best validation loss {best_val_loss:.4f}")
        else:
            print(f"No checkpoint found at {checkpoint_path}, starting from scratch")
    
    # Initialize TensorBoard writer
    log_dir = os.path.join('logs', 'ojr_' + time.strftime('%Y%m%d-%H%M%S'))
    writer = SummaryWriter(log_dir)
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        ojr_model.train()
        train_loss = 0.0
        train_mse = 0.0
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, data in pbar:
            try:
                # Handle data unpacking with error checking
                if not isinstance(data, tuple) or len(data) != 2:
                    logging.warning(f"Skipping batch {batch_idx}: Invalid data format")
                    continue
                
                frames, _ = data
                
                # Check for NaN or invalid values
                if torch.isnan(frames).any() or torch.isinf(frames).any():
                    logging.warning(f"Skipping batch {batch_idx}: NaN or Inf values in frames")
                    continue
                
                # Move data to device
                frames = frames.to(device)
                
                # Extract keypoints using SSHFD
                with torch.no_grad():
                    keypoints_seq, confidences_seq, _ = sshfd_model(frames)
                
                # Create artificial occlusions for training
                batch_size = keypoints_seq.size(0)
                last_frame_keypoints = keypoints_seq[:, -1, :]
                last_frame_confidences = confidences_seq[:, -1, :]
                
                # Create occlusion mask based on random selection
                # In a real implementation, you'd use more sophisticated methods
                # to simulate realistic occlusions
                random_occlusion = torch.rand(batch_size, config['model']['num_keypoints']).to(device)
                occlusion_mask = (random_occlusion < config['training']['occlusion_threshold']).float()
                
                # Expand mask for x,y coordinates
                expanded_mask = torch.zeros_like(last_frame_keypoints)
                expanded_mask[:, 0::2] = occlusion_mask  # x coordinates
                expanded_mask[:, 1::2] = occlusion_mask  # y coordinates
                
                # Apply occlusions (replace with zeros)
                occluded_keypoints = last_frame_keypoints.clone()
                occluded_keypoints[expanded_mask > 0] = 0
                
                # Create occluded sequence
                occluded_seq = keypoints_seq.clone()
                occluded_seq[:, -1, :] = occluded_keypoints
                
                # Forward pass
                recovered_keypoints = ojr_model(occluded_seq, expanded_mask)
                
                # Calculate loss only for occluded joints
                # Ground truth is the original keypoints from SSHFD
                loss = criterion(
                    recovered_keypoints[expanded_mask > 0],
                    last_frame_keypoints[expanded_mask > 0]
                )
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update metrics
                train_loss += loss.item()
                
                # Calculate MSE
                mse = torch.mean((recovered_keypoints[expanded_mask > 0] - last_frame_keypoints[expanded_mask > 0]) ** 2)
                train_mse += mse.item()
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item(), 'mse': mse.item()})
            except Exception as e:
                logging.error(f"Error in batch {batch_idx}: {str(e)}")
                continue  # Skip this batch and continue with the next one
        
        # Calculate average metrics
        train_loss /= len(train_loader)
        train_mse /= len(train_loader)
        
        # Validation phase
        ojr_model.eval()
        val_loss = 0.0
        val_mse = 0.0
        
        pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for batch_idx, data in pbar:
                try:
                    # Handle data unpacking with error checking
                    if not isinstance(data, tuple) or len(data) != 2:
                        logging.warning(f"Skipping validation batch {batch_idx}: Invalid data format")
                        continue
                    
                    frames, _ = data
                    
                    # Check for NaN or invalid values
                    if torch.isnan(frames).any() or torch.isinf(frames).any():
                        logging.warning(f"Skipping validation batch {batch_idx}: NaN or Inf values in frames")
                        continue
                    
                    # Move data to device
                    frames = frames.to(device)
                    
                    # Extract keypoints using SSHFD
                    keypoints_seq, confidences_seq, _ = sshfd_model(frames)
                    
                    # Create artificial occlusions for validation
                    batch_size = keypoints_seq.size(0)
                    last_frame_keypoints = keypoints_seq[:, -1, :]
                    last_frame_confidences = confidences_seq[:, -1, :]
                    
                    # Create occlusion mask
                    random_occlusion = torch.rand(batch_size, config['model']['num_keypoints']).to(device)
                    occlusion_mask = (random_occlusion < config['training']['occlusion_threshold']).float()
                    
                    # Expand mask for x,y coordinates
                    expanded_mask = torch.zeros_like(last_frame_keypoints)
                    expanded_mask[:, 0::2] = occlusion_mask
                    expanded_mask[:, 1::2] = occlusion_mask
                    
                    # Apply occlusions
                    occluded_keypoints = last_frame_keypoints.clone()
                    occluded_keypoints[expanded_mask > 0] = 0
                    
                    # Create occluded sequence
                    occluded_seq = keypoints_seq.clone()
                    occluded_seq[:, -1, :] = occluded_keypoints
                    
                    # Forward pass
                    recovered_keypoints = ojr_model(occluded_seq, expanded_mask)
                    
                    # Calculate loss only for occluded joints
                    loss = criterion(
                        recovered_keypoints[expanded_mask > 0],
                        last_frame_keypoints[expanded_mask > 0]
                    )
                    
                    # Update metrics
                    val_loss += loss.item()
                    
                    # Calculate MSE
                    mse = torch.mean((recovered_keypoints[expanded_mask > 0] - last_frame_keypoints[expanded_mask > 0]) ** 2)
                    val_mse += mse.item()
                    
                    # Update progress bar
                    pbar.set_postfix({'loss': loss.item(), 'mse': mse.item()})
                except Exception as e:
                    logging.error(f"Error in validation batch {batch_idx}: {str(e)}")
                    continue  # Skip this batch and continue with the next one
        
        # Calculate average metrics
        val_loss /= len(val_loader)
        val_mse /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, MSE: {train_mse:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, MSE: {val_mse:.4f}")
        
        # Write to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('MSE/train', train_mse, epoch)
        writer.add_scalar('MSE/val', val_mse, epoch)
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': ojr_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }, is_best,
        filename=os.path.join('weights', 'ojr_checkpoint.pth'),
        best_filename=config['model']['ojr_save_path'])
        
        if is_best:
            print("  Saved new best model!")
    
    writer.close()
    print("OJR Training completed!")
    
    return ojr_model


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create necessary directories
    os.makedirs('weights', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Train models
    sshfd_model = None
    
    if args.train_sshfd:
        sshfd_model = train_sshfd(config, args)
    
    if args.train_ojr:
        train_ojr(config, args, sshfd_model)


if __name__ == "__main__":
    main()