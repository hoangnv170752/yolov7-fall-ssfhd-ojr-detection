#!/usr/bin/env python
"""
Test script to check the data format expected by the model.
"""

import os
import torch
import yaml
import logging
from pathlib import Path
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_format_test.log"),
        logging.StreamHandler()
    ]
)

# Import project modules
from utils.data_utils import FallDetectionDataset

def main():
    """Main function."""
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dataset
    data_root = config['data']['root']
    train_list = config['data']['train_list']
    
    print(f"Creating dataset with root: {data_root}, train_list: {train_list}")
    
    try:
        # Create dataset with a small subset
        dataset = FallDetectionDataset(
            data_root=data_root,
            video_list=train_list,
            temporal_window=config['model']['temporal_window'],
            transform=None,
            mode='train'
        )
        
        print(f"Dataset created with {len(dataset)} samples")
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0  # Use single process for debugging
        )
        
        print("Dataloader created, attempting to load first batch...")
        
        # Try to load first batch
        for batch_idx, data in enumerate(dataloader):
            print(f"Batch {batch_idx} loaded successfully!")
            
            # Print data structure
            if isinstance(data, tuple) or isinstance(data, list):
                print(f"Data is a {type(data).__name__} with {len(data)} elements")
                
                for i, item in enumerate(data):
                    if isinstance(item, torch.Tensor):
                        print(f"  Item {i}: Tensor of shape {item.shape} and dtype {item.dtype}")
                    else:
                        print(f"  Item {i}: {type(item).__name__}")
            else:
                print(f"Data is a {type(data).__name__}")
                
                if isinstance(data, torch.Tensor):
                    print(f"  Shape: {data.shape}, dtype: {data.dtype}")
            
            # Only process first batch
            break
            
    except Exception as e:
        logging.error(f"Error creating or using dataset: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
