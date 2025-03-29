#!/usr/bin/env python
"""
Direct fix for the checkpoint saving issue.
This script directly patches the config and ensures the weights directory exists.
"""
import os
import yaml
import torch
from pathlib import Path

def fix_config():
    """Fix the config file to use proper paths."""
    config_path = "config/config.yaml"
    
    # Read the config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Check the save_path
    save_path = config.get("model", {}).get("save_path", "")
    if not save_path or not isinstance(save_path, str):
        print(f"Invalid save_path in config: {save_path}")
        # Set a proper path
        config["model"]["save_path"] = "weights/sshfd_model.pt"
        print(f"Updated save_path to: {config['model']['save_path']}")
    
    # Check ojr_save_path
    ojr_save_path = config.get("model", {}).get("ojr_save_path", "")
    if not ojr_save_path or not isinstance(ojr_save_path, str):
        print(f"Invalid ojr_save_path in config: {ojr_save_path}")
        # Set a proper path
        config["model"]["ojr_save_path"] = "weights/ojr_model.pt"
        print(f"Updated ojr_save_path to: {config['model']['ojr_save_path']}")
    
    # Write the updated config
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Updated config file: {config_path}")

def create_weights_dir():
    """Create the weights directory."""
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    print(f"Created weights directory: {weights_dir.absolute()}")
    
    # Create a test file to verify write permissions
    test_file = weights_dir / "test_file.txt"
    with open(test_file, "w") as f:
        f.write("Test file to verify write permissions")
    print(f"Created test file: {test_file}")
    
    # Try to create a test PyTorch checkpoint
    try:
        dummy_state = {"test": "data"}
        test_checkpoint = weights_dir / "test_checkpoint.pt"
        torch.save(dummy_state, test_checkpoint)
        print(f"Successfully created test checkpoint: {test_checkpoint}")
    except Exception as e:
        print(f"Error creating test checkpoint: {str(e)}")

def patch_train_script():
    """Create a patched version of the train.py script."""
    patched_script = """#!/usr/bin/env python
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
"""
    
    with open("patched_train.py", "w") as f:
        f.write(patched_script)
    
    # Make the script executable
    os.chmod("patched_train.py", 0o755)
    print("Created patched_train.py with torch.save monkey patch")

def main():
    """Main function."""
    print("Applying direct fix for checkpoint saving issue...")
    
    # Create weights directory
    create_weights_dir()
    
    # Fix the config
    fix_config()
    
    # Patch the train script
    patch_train_script()
    
    print("\nFixes applied successfully!")
    print("\nYou can now run training with:")
    print("python patched_train.py --train-sshfd --gpu -1")

if __name__ == "__main__":
    main()
