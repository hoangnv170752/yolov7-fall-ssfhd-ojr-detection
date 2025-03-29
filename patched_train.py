#!/usr/bin/env python
import os
import sys
import torch
from pathlib import Path

# Ensure weights directory exists
weights_dir = Path("weights")
weights_dir.mkdir(exist_ok=True)

# Monkey patch torch.save to handle directory creation
original_save = torch.save
def patched_save(obj, f, *args, **kwargs):
    if isinstance(f, str):
        os.makedirs(os.path.dirname(f), exist_ok=True)
    return original_save(obj, f, *args, **kwargs)
torch.save = patched_save

# Import the original training script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import main

if __name__ == "__main__":
    main()
