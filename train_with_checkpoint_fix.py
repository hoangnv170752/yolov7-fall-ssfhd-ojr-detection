#!/usr/bin/env python
import os
import sys
import torch
from pathlib import Path

# Ensure weights directory exists
weights_dir = Path("weights")
weights_dir.mkdir(exist_ok=True)

# Import the original training script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import main

if __name__ == "__main__":
    try:
        main()
    except AttributeError as e:
        if "expected 'f' to be string" in str(e):
            print("\nError: Failed to save checkpoint. Fixing the issue...")
            # Create weights directory if it doesn't exist
            weights_dir = Path("weights")
            weights_dir.mkdir(exist_ok=True)
            print(f"Created weights directory: {weights_dir.absolute()}")
            print("Please run the training again.")
        else:
            raise
