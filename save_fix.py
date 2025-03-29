#!/usr/bin/env python
"""
Fix for the save_checkpoint function in model_utils.py
"""
import os
import sys
import torch
from pathlib import Path

def main():
    # Create weights directory
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    print(f"Created weights directory: {weights_dir.absolute()}")
    
    # Create a simple checkpoint file to test saving
    test_file = weights_dir / "test_checkpoint.pt"
    test_data = {"test": "data"}
    torch.save(test_data, test_file)
    print(f"Successfully saved test checkpoint to: {test_file}")
    
    # Create a simple wrapper for save_checkpoint
    wrapper_code = """
# Monkey patch torch.save to handle directory creation
import os
import torch

original_save = torch.save

def patched_save(obj, f, *args, **kwargs):
    if isinstance(f, str):
        os.makedirs(os.path.dirname(f), exist_ok=True)
    return original_save(obj, f, *args, **kwargs)

torch.save = patched_save
"""
    
    # Add the wrapper to the beginning of train.py
    with open("train.py", "r") as f:
        train_content = f.read()
    
    # Check if already patched
    if "patched_save" in train_content:
        print("train.py is already patched.")
    else:
        # Insert after the first import
        import_end = train_content.find("import")
        import_end = train_content.find("\n", import_end)
        
        patched_content = train_content[:import_end+1] + wrapper_code + train_content[import_end+1:]
        
        # Write back
        with open("train.py", "w") as f:
            f.write(patched_content)
        
        print("Successfully patched train.py with torch.save wrapper.")
    
    print("\nFix applied. You can now run training with:")
    print("python train.py --train-sshfd --gpu -1")

if __name__ == "__main__":
    main()
