#!/usr/bin/env python
"""
Script to combine trained SSHFD and OJR models into a single unified model.
This creates a combined model for inference that includes both fall detection
and occluded joint recovery capabilities.
"""

import os
import torch
import argparse
import yaml
from pathlib import Path

from models.sshfd import SSHFD
from models.ojr import OJR
from models.model_utils import load_checkpoint

class UnifiedModel(torch.nn.Module):
    """
    Unified model that combines SSHFD and OJR capabilities.
    """
    def __init__(self, sshfd_model, ojr_model):
        super(UnifiedModel, self).__init__()
        self.sshfd_model = sshfd_model
        self.ojr_model = ojr_model
        
    def forward(self, frames, keypoints=None, occlusion_mask=None):
        """
        Forward pass through the unified model.
        
        Args:
            frames: Input video frames [batch_size, num_frames, channels, height, width]
            keypoints: Optional keypoints [batch_size, num_keypoints*2]
            occlusion_mask: Optional occlusion mask [batch_size, num_keypoints]
            
        Returns:
            Dictionary with fall detection and joint recovery outputs
        """
        # Process through SSHFD model
        sshfd_outputs = self.sshfd_model(frames)
        
        # If keypoints are not provided, use the ones from SSHFD
        if keypoints is None:
            keypoints = sshfd_outputs['keypoints']
        
        # Process through OJR model if occlusion is detected
        if occlusion_mask is not None and occlusion_mask.sum() > 0:
            recovered_keypoints = self.ojr_model(keypoints, occlusion_mask)
            # Replace occluded keypoints with recovered ones
            final_keypoints = keypoints.clone()
            final_keypoints[occlusion_mask.bool()] = recovered_keypoints[occlusion_mask.bool()]
        else:
            final_keypoints = keypoints
            
        # Combine outputs
        outputs = {
            'fall_prob': sshfd_outputs['fall_prob'],
            'keypoints': sshfd_outputs['keypoints'],
            'recovered_keypoints': final_keypoints
        }
        
        return outputs

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Combine SSHFD and OJR models')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--sshfd-weights', type=str, help='Path to SSHFD weights (default from config)')
    parser.add_argument('--ojr-weights', type=str, help='Path to OJR weights (default from config)')
    parser.add_argument('--output', type=str, default='weights/unified_model.pt', help='Output path for unified model')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU ID to use')
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load SSHFD model
    print("Loading SSHFD model...")
    sshfd_model = SSHFD(
        num_keypoints=config['model']['num_keypoints'],
        hidden_dim=config['model']['hidden_dim'],
        temporal_window=config['model']['temporal_window']
    ).to(device)
    
    sshfd_weights = args.sshfd_weights or config['model']['save_path']
    if os.path.exists(sshfd_weights):
        load_checkpoint(sshfd_model, sshfd_weights)
        print(f"Loaded SSHFD weights from {sshfd_weights}")
    else:
        print(f"Warning: SSHFD weights not found at {sshfd_weights}")
        return
    
    # Load OJR model
    print("Loading OJR model...")
    ojr_model = OJR(
        num_keypoints=config['model']['num_keypoints'],
        hidden_dim=config['model']['ojr_hidden_dim'],
        num_layers=config['model']['ojr_num_layers']
    ).to(device)
    
    ojr_weights = args.ojr_weights or config['model']['ojr_save_path']
    if os.path.exists(ojr_weights):
        load_checkpoint(ojr_model, ojr_weights)
        print(f"Loaded OJR weights from {ojr_weights}")
    else:
        print(f"Warning: OJR weights not found at {ojr_weights}")
        return
    
    # Create unified model
    print("Creating unified model...")
    unified_model = UnifiedModel(sshfd_model, ojr_model)
    
    # Save unified model
    torch.save({
        'model': unified_model.state_dict(),
        'config': config,
        'sshfd_path': sshfd_weights,
        'ojr_path': ojr_weights
    }, args.output)
    
    print(f"Unified model saved to {args.output}")
    print("\nTo use the unified model for inference:")
    print(f"python detect_falls.py --weights {args.output} --unified")

if __name__ == "__main__":
    main()
