#!/usr/bin/env python
"""
Training script for SSHFD and OJR models.

This script trains the SSHFD model for fall detection and the
OJR model for occluded joint recovery.
"""

import os
import torch
import numpy as np
import argparse
import yaml
import time
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
    
    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(config)
    
    # Initialize model
    model = SSHFD(
        num_keypoints=config['model']['num_keypoints'],
        hidden_dim=config['model']['hidden_dim'],
        temporal_window=config['model']['temporal_window']
    )
    
    # Initialize weights
    init_weights(model, init_type='xavier')
    
    # Move model to device
    model = model.to(device)
    
    # Define optimizer and schedulers
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Loss functions
    keypoint_criterion = torch.nn.MSELoss()
    fall_criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, config['training']['fall_class_weight']]).to(device)
    )
    
    # Initialize training variables
    start_epoch = 0
    best_val_loss = float('inf')
    num_epochs = config['training']['epochs']
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint_path = config['model']['save_path']
        if os.path.exists(checkpoint_path):
            start_epoch, best_val_loss = load_checkpoint(model, optimizer, checkpoint_path)
            print(f"Resuming from epoch {start_epoch} with best validation loss {best_val_loss:.4f}")
        else:
            print(f"No checkpoint found at {checkpoint_path}, starting from scratch")
    
    # Initialize TensorBoard writer
    log_dir = os.path.join('logs', time.strftime('%Y%m%d-%H%M%S'))
    writer = SummaryWriter(log_dir)
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_keypoint_loss = 0.0
        train_fall_loss = 0.0
        train_acc = 0.0
        train_predictions = []
        train_targets = []
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, (frames, labels) in pbar:
            # Move data to device
            frames = frames.to(device)
            labels = labels.to(device)
            
            # Forward pass
            keypoints, confidences, fall_preds = model(frames)
            
            # For keypoint loss, we would need ground truth keypoints
            # In a real implementation, you'd use actual GT keypoints
            # Here we use a simplified approach with dummy targets
            # In a real scenario, your dataset would provide GT keypoints
            keypoint_gt = torch.zeros_like(keypoints).to(device)
            confidence_gt = torch.ones_like(confidences).to(device)
            
            # Calculate losses
            kp_loss = keypoint_criterion(keypoints, keypoint_gt)
            fall_loss = fall_criterion(fall_preds, labels)
            
            # Combined loss
            loss = kp_loss * config['training']['keypoint_loss_weight'] + \
                   fall_loss * config['training']['fall_loss_weight']
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_keypoint_loss += kp_loss.item()
            train_fall_loss += fall_loss.item()
            
            # Compute accuracy
            batch_acc = compute_accuracy(fall_preds, labels)
            train_acc += batch_acc
            
            # Store predictions and targets for precision, recall, F1
            preds = torch.softmax(fall_preds, dim=1).detach().cpu().numpy()
            train_predictions.extend(preds[:, 1])  # Probability of fall
            train_targets.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'kp_loss': kp_loss.item(),
                'fall_loss': fall_loss.item(),
                'acc': batch_acc
            })
        
        # Calculate average metrics
        train_loss /= len(train_loader)
        train_keypoint_loss /= len(train_loader)
        train_fall_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        # Calculate precision, recall, F1
        train_precision, train_recall, train_f1 = compute_precision_recall_f1(
            train_predictions, train_targets, threshold=0.5
        )
        
        # Calculate AUC
        train_auc = compute_auc(train_predictions, train_targets)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_keypoint_loss = 0.0
        val_fall_loss = 0.0
        val_acc = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for frames, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                # Move data to device
                frames = frames.to(device)
                labels = labels.to(device)
                
                # Forward pass
                keypoints, confidences, fall_preds = model(frames)
                
                # Dummy keypoint targets
                keypoint_gt = torch.zeros_like(keypoints).to(device)
                confidence_gt = torch.ones_like(confidences).to(device)
                
                # Calculate losses
                kp_loss = keypoint_criterion(keypoints, keypoint_gt)
                fall_loss = fall_criterion(fall_preds, labels)
                
                # Combined loss
                loss = kp_loss * config['training']['keypoint_loss_weight'] + \
                       fall_loss * config['training']['fall_loss_weight']
                
                # Update metrics
                val_loss += loss.item()
                val_keypoint_loss += kp_loss.item()
                val_fall_loss += fall_loss.item()
                
                # Compute accuracy
                val_acc += compute_accuracy(fall_preds, labels)
                
                # Store predictions and targets
                preds = torch.softmax(fall_preds, dim=1).cpu().numpy()
                val_predictions.extend(preds[:, 1])
                val_targets.extend(labels.cpu().numpy())
        
        # Calculate average metrics
        val_loss /= len(val_loader)
        val_keypoint_loss /= len(val_loader)
        val_fall_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        # Calculate precision, recall, F1
        val_precision, val_recall, val_f1 = compute_precision_recall_f1(
            val_predictions, val_targets, threshold=0.5
        )
        
        # Calculate AUC
        val_auc = compute_auc(val_predictions, val_targets)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f} (KP: {train_keypoint_loss:.4f}, Fall: {train_fall_loss:.4f})")
        print(f"  Train Metrics: Acc={train_acc:.4f}, P={train_precision:.4f}, R={train_recall:.4f}, F1={train_f1:.4f}, AUC={train_auc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} (KP: {val_keypoint_loss:.4f}, Fall: {val_fall_loss:.4f})")
        print(f"  Val Metrics: Acc={val_acc:.4f}, P={val_precision:.4f}, R={val_recall:.4f}, F1={val_f1:.4f}, AUC={val_auc:.4f}")
        
        # Write to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('F1/train', train_f1, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)
        writer.add_scalar('AUC/train', train_auc, epoch)
        writer.add_scalar('AUC/val', val_auc, epoch)
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }, is_best, 
        filename=os.path.join('weights', 'sshfd_checkpoint.pth'),
        best_filename=config['model']['save_path'])
        
        if is_best:
            print("  Saved new best model!")
    
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
        sshfd_model.load_state_dict(torch.load(config['model']['save_path'], map_location=device))
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
        lr=config['training']['ojr_lr'],
        weight_decay=config['training']['weight_decay']
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
        for batch_idx, (frames, _) in pbar:
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
            # In a real implementation, you might use more sophisticated methods
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
        
        # Calculate average metrics
        train_loss /= len(train_loader)
        train_mse /= len(train_loader)
        
        # Validation phase
        ojr_model.eval()
        val_loss = 0.0
        val_mse = 0.0
        
        with torch.no_grad():
            for frames, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
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