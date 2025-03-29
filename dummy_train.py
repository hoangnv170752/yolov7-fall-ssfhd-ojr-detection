#!/usr/bin/env python
"""
Dummy training script for SSHFD model.

This script creates a simplified version of the training process
using dummy data to verify that the model architecture works.
"""

import os
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
from models.model_utils import init_weights, load_checkpoint
from utils.metrics import compute_accuracy, compute_precision_recall_f1, compute_auc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dummy_training.log"),
        logging.StreamHandler()
    ]
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train SSHFD model with dummy data')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU ID to use')
    
    return parser.parse_args()

class DummyDataset(torch.utils.data.Dataset):
    """Dummy dataset that generates random data for training."""
    
    def __init__(self, num_samples=100, temporal_window=5, num_keypoints=17):
        self.num_samples = num_samples
        self.temporal_window = temporal_window
        self.num_keypoints = num_keypoints
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random frames (temporal_window, 3, 224, 224)
        frames = torch.rand(self.temporal_window, 3, 224, 224)
        
        # Generate random label (0: no fall, 1: fall)
        label = torch.randint(0, 2, (1,)).item()
        
        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)
        
        return frames, label

def train_sshfd_with_dummy_data(config, args):
    """
    Train SSHFD model with dummy data.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    print("=== Training SSHFD Model with Dummy Data ===")
    
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
    fall_criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, config['training']['fall_class_weight']]).to(device)
    )
    
    # Define optimizer
    lr = config['training']['lr']
    weight_decay = config['training']['weight_decay']
    print(f"Learning rate value: {lr} (type: {type(lr)})")
    print(f"Weight decay value: {weight_decay} (type: {type(weight_decay)})")
    
    # Convert to float if needed
    if isinstance(weight_decay, str):
        weight_decay = float(weight_decay)
    print(f"After conversion - Weight decay: {weight_decay} (type: {type(weight_decay)})")
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training parameters
    num_epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    
    # Create dummy datasets
    train_dataset = DummyDataset(
        num_samples=100,
        temporal_window=config['model']['temporal_window'],
        num_keypoints=config['model']['num_keypoints']
    )
    
    val_dataset = DummyDataset(
        num_samples=20,
        temporal_window=config['model']['temporal_window'],
        num_keypoints=config['model']['num_keypoints']
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1
    )
    
    # Initialize training variables
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Initialize TensorBoard writer
    log_dir = os.path.join('logs', 'dummy_' + time.strftime('%Y%m%d-%H%M%S'))
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
        for batch_idx, data in pbar:
            try:
                # Handle data unpacking with error checking
                if not isinstance(data, tuple) or len(data) != 2:
                    logging.warning(f"Skipping batch {batch_idx}: Invalid data format")
                    continue
                
                frames, labels = data
                
                # Check for NaN or invalid values
                if torch.isnan(frames).any() or torch.isinf(frames).any():
                    logging.warning(f"Skipping batch {batch_idx}: NaN or Inf values in frames")
                    continue
                
                # Move data to device
                frames = frames.to(device)
                labels = labels.to(device)
                
                # Forward pass
                keypoints, confidences, fall_preds = model(frames)
                
                # For keypoint loss, we would need ground truth keypoints
                # Here we use a simplified approach with dummy targets
                keypoint_gt = torch.zeros_like(keypoints).to(device)
                confidence_gt = torch.ones_like(confidences).to(device)
                
                # Calculate losses
                kp_loss = keypoint_criterion(keypoints, keypoint_gt)
                fall_loss = fall_criterion(fall_preds, labels)
                
                # Combined loss
                loss = kp_loss * config['training']['keypoint_loss_weight'] + \
                       fall_loss * config['training']['fall_loss_weight']
                
                # Check for NaN losses
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    logging.warning(f"Skipping batch {batch_idx}: NaN or Inf loss values")
                    continue
                
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
            except Exception as e:
                logging.error(f"Error in batch {batch_idx}: {str(e)}")
                continue  # Skip this batch and continue with the next one
        
        # Check if we processed any batches
        if len(train_loader) == 0:
            logging.error(f"No batches processed in epoch {epoch+1}. Skipping epoch.")
            continue
            
        # Calculate average metrics (avoid division by zero)
        processed_batches = max(1, len(train_loader))
        train_loss /= processed_batches
        train_keypoint_loss /= processed_batches
        train_fall_loss /= processed_batches
        train_acc /= processed_batches
        
        # Calculate precision, recall, F1
        if len(train_predictions) > 0 and len(train_targets) > 0:
            train_precision, train_recall, train_f1 = compute_precision_recall_f1(
                train_predictions, train_targets, threshold=0.5
            )
            # Calculate AUC
            train_auc = compute_auc(train_predictions, train_targets)
        else:
            logging.warning(f"No valid predictions in epoch {epoch+1}")
            train_precision, train_recall, train_f1, train_auc = 0, 0, 0, 0
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_keypoint_loss = 0.0
        val_fall_loss = 0.0
        val_acc = 0.0
        val_predictions = []
        val_targets = []
        
        pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for batch_idx, data in pbar:
                try:
                    # Handle data unpacking with error checking
                    if not isinstance(data, tuple) or len(data) != 2:
                        logging.warning(f"Skipping validation batch {batch_idx}: Invalid data format")
                        continue
                    
                    frames, labels = data
                    
                    # Check for NaN or invalid values
                    if torch.isnan(frames).any() or torch.isinf(frames).any():
                        logging.warning(f"Skipping validation batch {batch_idx}: NaN or Inf values in frames")
                        continue
                    
                    # Move data to device
                    frames = frames.to(device)
                    labels = labels.to(device)
                    
                    # Forward pass
                    keypoints, confidences, fall_preds = model(frames)
                    
                    # For keypoint loss, we would need ground truth keypoints
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
                    batch_acc = compute_accuracy(fall_preds, labels)
                    val_acc += batch_acc
                    
                    # Store predictions and targets for precision, recall, F1
                    preds = torch.softmax(fall_preds, dim=1).detach().cpu().numpy()
                    val_predictions.extend(preds[:, 1])  # Probability of fall
                    val_targets.extend(labels.cpu().numpy())
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': loss.item(),
                        'kp_loss': kp_loss.item(),
                        'fall_loss': fall_loss.item(),
                        'acc': batch_acc
                    })
                except Exception as e:
                    logging.error(f"Error in validation batch {batch_idx}: {str(e)}")
                    continue  # Skip this batch and continue with the next one
        
        # Check if we processed any validation batches
        if len(val_loader) == 0:
            logging.error(f"No validation batches processed in epoch {epoch+1}. Skipping validation.")
            continue
            
        # Calculate average metrics (avoid division by zero)
        processed_val_batches = max(1, len(val_loader))
        val_loss /= processed_val_batches
        val_keypoint_loss /= processed_val_batches
        val_fall_loss /= processed_val_batches
        val_acc /= processed_val_batches
        
        # Calculate precision, recall, F1
        if len(val_predictions) > 0 and len(val_targets) > 0:
            val_precision, val_recall, val_f1 = compute_precision_recall_f1(
                val_predictions, val_targets, threshold=0.5
            )
            # Calculate AUC
            val_auc = compute_auc(val_predictions, val_targets)
        else:
            logging.warning(f"No valid validation predictions in epoch {epoch+1}")
            val_precision, val_recall, val_f1, val_auc = 0, 0, 0, 0
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('F1/train', train_f1, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)
        writer.add_scalar('AUC/train', train_auc, epoch)
        writer.add_scalar('AUC/val', val_auc, epoch)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f} (KP: {train_keypoint_loss:.4f}, Fall: {train_fall_loss:.4f})")
        print(f"  Train Metrics: Acc={train_acc:.4f}, P={train_precision:.4f}, R={train_recall:.4f}, F1={train_f1:.4f}, AUC={train_auc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} (KP: {val_keypoint_loss:.4f}, Fall: {val_fall_loss:.4f})")
        print(f"  Val Metrics: Acc={val_acc:.4f}, P={val_precision:.4f}, R={val_recall:.4f}, F1={val_f1:.4f}, AUC={val_auc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, config['model']['save_path'])
            print("  Saved new best model!")
    
    print("SSHFD Training completed!")
    return model

def main():
    """Main function."""
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Train SSHFD model with dummy data
    sshfd_model = train_sshfd_with_dummy_data(config, args)

if __name__ == "__main__":
    main()
