
import torch
import os

# Create weights directory if it doesn't exist
os.makedirs('weights', exist_ok=True)

# Process SSHFD checkpoint
try:
    checkpoint = torch.load('weights/sshfd_best.pt', map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        torch.save(state_dict, 'weights/sshfd_state_dict.pt')
        print("Extracted SSHFD state dictionary saved to weights/sshfd_state_dict.pt")
    else:
        print("SSHFD checkpoint already contains only the state dictionary")
except Exception as e:
    print(f"Error processing SSHFD checkpoint: {e}")

# Process OJR checkpoint
try:
    checkpoint = torch.load('weights/ojr_best.pt', map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        torch.save(state_dict, 'weights/ojr_state_dict.pt')
        print("Extracted OJR state dictionary saved to weights/ojr_state_dict.pt")
    else:
        print("OJR checkpoint already contains only the state dictionary")
except Exception as e:
    print(f"Error processing OJR checkpoint: {e}")
