#!/usr/bin/env python
"""
Evaluation script for SSHFD and OJR models.

This script evaluates the trained SSHFD and OJR models on the test set
and generates visualizations to analyze their performance.
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
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# Import project modules
from models.sshfd import SSHFD
from models.ojr import OJR
from models.model_utils import load_yolov7_model
from utils.data_utils import create_dataloaders
from utils.metrics import evaluate_fall_detection, evaluate_ojr
from utils.visual_utils import visualize_fall_detection, visualize_occlusion_recovery


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate SSHFD and OJR models')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--sshfd-only', action='store_true', help='Evaluate only SSHFD model')
    parser.add_argument('--ojr-only', action='store_true', help='Evaluate only OJR model')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--output-dir', type=str, default='outputs/evaluation', help='Output directory for results')
    
    return parser.parse_args()


def evaluate_sshfd(config, args):
    """
    Evaluate SSHFD model.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
        
    Returns:
        metrics: Evaluation metrics
    """
    print("=== Evaluating SSHFD Model ===")
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    print(f"Using device: {device}")
    
    # Create dataloaders
    _, _, test_loader = create_dataloaders(config)
    
    # Load model
    model = SSHFD(
        num_keypoints=config['model']['num_keypoints'],
        hidden_dim=config['model']['hidden_dim'],
        temporal_window=config['model']['temporal_window']
    )
    
    # Fix for loading the checkpoint correctly
    checkpoint = torch.load(config['model']['save_path'], map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device).eval()
    
    # Evaluate
    print("Evaluating on test set...")
    metrics = evaluate_fall_detection(model, test_loader, device)
    
    # Print results
    print("\nResults:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")
    
    # Print confusion matrix
    cm = metrics['confusion_matrix']
    print("\nConfusion Matrix:")
    
    # Handle case where only one class is present
    if cm.shape == (1, 1):
        # Check which class it is (only one is present)
        with torch.no_grad():
            for frames, labels in test_loader:
                all_targets = labels.numpy()
                break
        
        only_class = 0 if all_targets[0] == 0 else 1
        if only_class == 0:
            print(f"  TN: {cm[0, 0]}, FP: 0")
            print(f"  FN: 0, TP: 0")
        else:
            print(f"  TN: 0, FP: 0")
            print(f"  FN: 0, TP: {cm[0, 0]}")
        print("Warning: Test set contains only one class. Consider adding examples of both classes.")
    else:
        # Normal case with both classes
        print(f"  TN: {cm[0, 0]}, FP: {cm[0, 1]}")
        print(f"  FN: {cm[1, 0]}, TP: {cm[1, 1]}")
    
    # Visualize results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate ROC curve
    visualize_roc_curve(test_loader, model, device, os.path.join(args.output_dir, 'roc_curve.png'))
    
    # Generate PR curve
    visualize_pr_curve(test_loader, model, device, os.path.join(args.output_dir, 'pr_curve.png'))
    
    # Generate confusion matrix visualization
    visualize_confusion_matrix(cm, os.path.join(args.output_dir, 'confusion_matrix.png'))
    
    # Save metrics
    save_metrics(metrics, os.path.join(args.output_dir, 'sshfd_metrics.txt'))
    
    return metrics


def evaluate_ojr_model(config, args, sshfd_model=None):
    """
    Evaluate OJR model.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
        sshfd_model: Pre-trained SSHFD model
        
    Returns:
        metrics: Evaluation metrics
    """
    print("=== Evaluating OJR Model ===")
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    print(f"Using device: {device}")
    
    # Create dataloaders
    _, _, test_loader = create_dataloaders(config)
    
    # Load SSHFD model if not provided
    if sshfd_model is None:
        sshfd_model = SSHFD(
            num_keypoints=config['model']['num_keypoints'],
            hidden_dim=config['model']['hidden_dim'],
            temporal_window=config['model']['temporal_window']
        )
        
        # Fix for loading the checkpoint correctly
        checkpoint = torch.load(config['model']['save_path'], map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            sshfd_model.load_state_dict(checkpoint['state_dict'])
        else:
            sshfd_model.load_state_dict(checkpoint)
            
        sshfd_model = sshfd_model.to(device).eval()

    # Load OJR model
    ojr_model = OJR(
        input_dim=config['model']['num_keypoints'] * 2,
        hidden_dim=config['model']['ojr_hidden_dim'],
        num_layers=config['model']['ojr_num_layers']
    )

    # Fix for loading the OJR checkpoint correctly
    checkpoint = torch.load(config['model']['ojr_save_path'], map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        ojr_model.load_state_dict(checkpoint['state_dict'])
    else:
        ojr_model.load_state_dict(checkpoint)
        
    ojr_model = ojr_model.to(device).eval()
    
    # Evaluate
    print("Evaluating on test set...")
    metrics = evaluate_ojr(ojr_model, sshfd_model, test_loader, device)
    
    # Print results
    print("\nResults:")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  Recovery PCK: {metrics['recovery_pck']:.4f}")
    
    # Visualize results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate occlusion recovery visualizations
    generate_occlusion_visualizations(sshfd_model, ojr_model, test_loader, device, args.output_dir)
    
    # Save metrics
    save_metrics(metrics, os.path.join(args.output_dir, 'ojr_metrics.txt'))
    
    return metrics


def visualize_roc_curve(dataloader, model, device, output_path):
    """
    Visualize ROC curve.
    
    Args:
        dataloader: Test dataloader
        model: SSHFD model
        device: PyTorch device
        output_path: Output path for the visualization
    """
    # Get all predictions and targets
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for frames, labels in tqdm(dataloader, desc="Generating ROC curve"):
            # Move to device
            frames = frames.to(device)
            labels = labels.to(device)
            
            # Forward pass
            _, _, fall_preds = model(frames)
            
            # Get probabilities
            probs = torch.softmax(fall_preds, dim=1)[:, 1]  # Probability of fall
            
            # Store predictions and targets
            all_predictions.extend(probs.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    # Check if both classes are present
    unique_classes = np.unique(all_targets)
    if len(unique_classes) < 2:
        print(f"Warning: Only one class ({unique_classes[0]}) present in test set. ROC curve requires both classes.")
        # Create empty plot with warning
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, f"ROC curve not available: Only class {unique_classes[0]} present in test set",
                horizontalalignment='center', verticalalignment='center', fontsize=12)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Not Available)')
        plt.savefig(output_path)
        plt.close()
        return
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(all_targets, all_predictions)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # Save figure
    plt.savefig(output_path)
    plt.close()
    
    print(f"ROC curve saved to {output_path}")


def visualize_pr_curve(dataloader, model, device, output_path):
    """
    Visualize Precision-Recall curve.
    
    Args:
        dataloader: Test dataloader
        model: SSHFD model
        device: PyTorch device
        output_path: Output path for the visualization
    """
    # Get all predictions and targets
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for frames, labels in tqdm(dataloader, desc="Generating PR curve"):
            # Move to device
            frames = frames.to(device)
            labels = labels.to(device)
            
            # Forward pass
            _, _, fall_preds = model(frames)
            
            # Get probabilities
            probs = torch.softmax(fall_preds, dim=1)[:, 1]  # Probability of fall
            
            # Store predictions and targets
            all_predictions.extend(probs.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    # Check if both classes are present
    unique_classes = np.unique(all_targets)
    if len(unique_classes) < 2:
        print(f"Warning: Only one class ({unique_classes[0]}) present in test set. PR curve requires both classes.")
        # Create empty plot with warning
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, f"PR curve not available: Only class {unique_classes[0]} present in test set",
                horizontalalignment='center', verticalalignment='center', fontsize=12)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (Not Available)')
        plt.savefig(output_path)
        plt.close()
        return
    
    # Calculate PR curve
    precision, recall, thresholds = precision_recall_curve(all_targets, all_predictions)
    pr_auc = auc(recall, precision)
    
    # Plot PR curve
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    
    # Save figure
    plt.savefig(output_path)
    plt.close()
    
    print(f"PR curve saved to {output_path}")


def visualize_confusion_matrix(cm, output_path):
    """
    Visualize confusion matrix.
    
    Args:
        cm: Confusion matrix
        output_path: Output path for the visualization
    """
    plt.figure(figsize=(8, 6))
    
    # Handle the case where only one class is present
    if cm.shape == (1, 1):
        # Create a 2x2 matrix for visualization
        expanded_cm = np.zeros((2, 2), dtype=int)
        
        # Determine which class it is (assuming we can tell from cm)
        # We don't know which class it is from cm alone, so we'll add a note
        expanded_cm[0, 0] = cm[0, 0]  # Place the value in either TN or TP position
        cm = expanded_cm
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix (Only one class present)')
        note = "Note: Only one class present in test set\nMatrix expanded to 2x2 for visualization"
        plt.figtext(0.5, 0.01, note, wrap=True, horizontalalignment='center', fontsize=9)
    else:
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
    
    plt.colorbar()
    
    classes = ['No Fall', 'Fall']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save figure
    plt.savefig(output_path)
    plt.close()
    
    print(f"Confusion matrix saved to {output_path}")


def generate_occlusion_visualizations(sshfd_model, ojr_model, dataloader, device, output_dir):
    """
    Generate OJR visualization examples.
    
    Args:
        sshfd_model: SSHFD model
        ojr_model: OJR model
        dataloader: Test dataloader
        device: PyTorch device
        output_dir: Output directory
    """
    occlusion_dir = os.path.join(output_dir, 'occlusion_recovery')
    os.makedirs(occlusion_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, (frames, _) in enumerate(dataloader):
            if i >= 5:  # Generate only 5 examples
                break
            
            # Move to device
            frames = frames.to(device)
            
            # Extract keypoints using SSHFD
            keypoints_seq, confidences_seq, _ = sshfd_model(frames)
            
            # Create artificial occlusions
            batch_size = keypoints_seq.size(0)
            last_frame_keypoints = keypoints_seq[:, -1, :]
            
            # Create random occlusion mask
            random_occlusion = torch.rand(batch_size, sshfd_model.num_keypoints).to(device)
            occlusion_mask = (random_occlusion < 0.3).float()  # 30% occlusion rate
            
            # Expand mask
            expanded_mask = torch.zeros_like(last_frame_keypoints)
            expanded_mask[:, 0::2] = occlusion_mask
            expanded_mask[:, 1::2] = occlusion_mask
            
            # Apply occlusions
            occluded_keypoints = last_frame_keypoints.clone()
            occluded_keypoints[expanded_mask > 0] = 0
            
            # Create occluded sequence
            occluded_seq = keypoints_seq.clone()
            occluded_seq[:, -1, :] = occluded_keypoints
            
            # Recover occluded keypoints
            recovered_keypoints = ojr_model(occluded_seq, expanded_mask)
            
            # Visualize recovery for the first sample in batch
            visualize_occlusion_recovery(
                frames[0, -1].cpu().numpy().transpose(1, 2, 0),  # Last frame
                frames[0, -1].cpu().numpy().transpose(1, 2, 0),  # Same frame for after
                occluded_keypoints[0].cpu().numpy(),
                recovered_keypoints[0].cpu().numpy(),
                expanded_mask[0].cpu().numpy(),
                os.path.join(occlusion_dir, f'recovery_example_{i}.png')
            )
    
    print(f"Occlusion recovery visualizations saved to {occlusion_dir}")


def save_metrics(metrics, output_path):
    """
    Save metrics to file.
    
    Args:
        metrics: Dictionary of metrics
        output_path: Output path
    """
    with open(output_path, 'w') as f:
        for key, value in metrics.items():
            if key == 'confusion_matrix':
                f.write(f"{key}:\n")
                cm = value
                # Handle case where only one class is present
                if cm.shape == (1, 1):
                    f.write(f"  Note: Only one class present in test set\n")
                    f.write(f"  Value: {cm[0, 0]}\n")
                else:
                    f.write(f"  TN: {cm[0, 0]}, FP: {cm[0, 1]}\n")
                    f.write(f"  FN: {cm[1, 0]}, TP: {cm[1, 1]}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    print(f"Metrics saved to {output_path}")


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate models
    if not args.ojr_only:
        sshfd_metrics = evaluate_sshfd(config, args)
    
    if not args.sshfd_only:
        # Load SSHFD model if needed for OJR evaluation
        if not args.ojr_only:
            device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
            sshfd_model = SSHFD(
                num_keypoints=config['model']['num_keypoints'],
                hidden_dim=config['model']['hidden_dim'],
                temporal_window=config['model']['temporal_window']
            )
            # Fix for loading the checkpoint correctly
            checkpoint = torch.load(config['model']['save_path'], map_location=device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                sshfd_model.load_state_dict(checkpoint['state_dict'])
            else:
                sshfd_model.load_state_dict(checkpoint)
            sshfd_model = sshfd_model.to(device).eval()
        else:
            sshfd_model = None
        
        ojr_metrics = evaluate_ojr_model(config, args, sshfd_model)


if __name__ == "__main__":
    main()