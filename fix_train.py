#!/usr/bin/env python
"""
Fix for training script to ensure proper checkpoint saving.
This script patches the train.py file to fix the checkpoint saving issue.
"""
import os
import sys
import torch
import yaml
import shutil
from pathlib import Path

def ensure_weights_directory():
    """Create weights directory if it doesn't exist."""
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    print(f"Created weights directory: {weights_dir.absolute()}")
    return weights_dir

def fix_save_checkpoint():
    """Fix the save_checkpoint function in model_utils.py."""
    # Create a backup of the original file
    model_utils_path = Path("models/model_utils.py")
    backup_path = model_utils_path.with_suffix(".py.bak")
    
    if not model_utils_path.exists():
        print(f"Error: {model_utils_path} does not exist.")
        return False
    
    # Create backup if it doesn't exist
    if not backup_path.exists():
        shutil.copy(model_utils_path, backup_path)
        print(f"Created backup of {model_utils_path} at {backup_path}")
    
    # Read the file content
    with open(model_utils_path, "r") as f:
        content = f.read()
    
    # Check if the function already contains the directory creation code
    if "os.makedirs(os.path.dirname(filename), exist_ok=True)" in content:
        print("The save_checkpoint function already contains the fix.")
        return True
    
    # Find the save_checkpoint function
    if "def save_checkpoint(" not in content:
        print("Error: save_checkpoint function not found in model_utils.py")
        return False
    
    # Add the directory creation code
    modified_content = content.replace(
        "def save_checkpoint(model, filename, optimizer=None, epoch=None):",
        "def save_checkpoint(model, filename, optimizer=None, epoch=None):\n    # Create directory if it doesn't exist\n    os.makedirs(os.path.dirname(filename), exist_ok=True)"
    )
    
    # Add import os if not already present
    if "import os" not in content:
        modified_content = "import os\n" + modified_content
    
    # Write the modified content back to the file
    with open(model_utils_path, "w") as f:
        f.write(modified_content)
    
    print(f"Fixed save_checkpoint function in {model_utils_path}")
    return True

def create_wrapper_script():
    """Create a wrapper script to run the training with proper error handling."""
    wrapper_path = Path("train_with_checkpoint_fix.py")
    
    with open(wrapper_path, "w") as f:
        f.write("""#!/usr/bin/env python
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
            print("\\nError: Failed to save checkpoint. Fixing the issue...")
            # Create weights directory if it doesn't exist
            weights_dir = Path("weights")
            weights_dir.mkdir(exist_ok=True)
            print(f"Created weights directory: {weights_dir.absolute()}")
            print("Please run the training again.")
        else:
            raise
""")
    
    # Make the script executable
    os.chmod(wrapper_path, 0o755)
    print(f"Created wrapper script at {wrapper_path}")
    return wrapper_path

def main():
    """Main function."""
    print("Fixing training checkpoint issues...")
    
    # Ensure weights directory exists
    weights_dir = ensure_weights_directory()
    
    # Fix the save_checkpoint function
    fix_save_checkpoint()
    
    # Create wrapper script
    wrapper_path = create_wrapper_script()
    
    print("\nFixes applied successfully!")
    print("\nYou can now run training with either:")
    print("1. python train.py --train-sshfd --gpu -1")
    print("   or")
    print("2. python train_with_checkpoint_fix.py --train-sshfd --gpu -1 (recommended)")

if __name__ == "__main__":
    main()
