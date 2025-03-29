import os
import torch
from pathlib import Path

def save_checkpoint(model, filename, optimizer=None, epoch=None):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        filename: Path to save checkpoint
        optimizer: Optimizer (optional)
        epoch: Current epoch (optional)
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Prepare state dict
    state = {
        'model': model.state_dict(),
        'epoch': epoch if epoch is not None else 0,
    }
    if optimizer is not None:
        state['optimizer'] = optimizer.state_dict()
    
    # Save checkpoint
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

if __name__ == "__main__":
    # Create weights directory if it doesn't exist
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    
    # Print confirmation
    print(f"Created weights directory: {weights_dir.absolute()}")
    print("You can now run training with: python train.py --train-sshfd --gpu -1")
