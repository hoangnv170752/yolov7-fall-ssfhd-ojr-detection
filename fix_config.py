#!/usr/bin/env python
"""
Fix for config.yaml formatting issues when running in Google Colab
"""
import os
import yaml
from pathlib import Path

def fix_config_yaml():
    """Fix formatting issues in config.yaml"""
    config_path = Path("config/config.yaml")
    
    # Read the current config
    with open(config_path, "r") as f:
        content = f.read()
    
    # Parse the config to check for errors
    try:
        config = yaml.safe_load(content)
        print("Config file is valid YAML.")
    except yaml.YAMLError as e:
        print(f"Found YAML error: {str(e)}")
        
        # Fix common issues
        # 1. Remove any trailing whitespace
        lines = content.splitlines()
        fixed_lines = [line.rstrip() for line in lines]
        
        # 2. Ensure proper indentation (2 spaces)
        for i in range(1, len(fixed_lines)):
            if fixed_lines[i].strip() and not fixed_lines[i].startswith(" "):
                if ":" in fixed_lines[i]:
                    # This is a top-level key, which is fine
                    pass
                else:
                    # This might be a continuation line missing indentation
                    fixed_lines[i] = "  " + fixed_lines[i]
        
        # 3. Ensure all keys have values (add colon if missing)
        for i in range(len(fixed_lines)):
            line = fixed_lines[i].strip()
            if line and not line.startswith("#") and ":" not in line:
                if i < len(fixed_lines) - 1 and fixed_lines[i+1].startswith("  "):
                    # This looks like a key without a colon
                    fixed_lines[i] = fixed_lines[i] + ":"
        
        # Join lines back together
        fixed_content = "\n".join(fixed_lines)
        
        # Create backup
        backup_path = config_path.with_suffix(".yaml.bak")
        with open(backup_path, "w") as f:
            f.write(content)
        print(f"Created backup at {backup_path}")
        
        # Write fixed content
        with open(config_path, "w") as f:
            f.write(fixed_content)
        
        # Verify the fix
        try:
            with open(config_path, "r") as f:
                yaml.safe_load(f)
            print("Fixed config file is now valid YAML.")
        except yaml.YAMLError as e:
            print(f"Warning: Config file still has YAML errors: {str(e)}")
            print("Manual inspection may be required.")

def create_colab_config():
    """Create a clean config file for Colab"""
    # Define a clean, minimal config
    config = {
        "data": {
            "img_size": 416,
            "root": "data",
            "test_list": "data/test_list_balanced_filtered.txt",
            "train_list": "data/train_list_balanced_filtered.txt",
            "val_list": "data/val_list_balanced_filtered.txt"
        },
        "inference": {
            "display": True,
            "gpu_id": 0,
            "output_fps": 30,
            "save_output": True,
            "video_input": True
        },
        "model": {
            "conf_thres": 0.25,
            "hidden_dim": 192,
            "iou_thres": 0.45,
            "num_keypoints": 17,
            "ojr_hidden_dim": 96,
            "ojr_num_layers": 2,
            "ojr_save_path": "weights/ojr_state_dict.pt",
            "save_path": "weights/sshfd_state_dict.pt",
            "temporal_window": 5,
            "yolo_weights": "weights/yolov7.pt"
        },
        "training": {
            "batch_size": 8,
            "epochs": 1,
            "fall_class_weight": 3.0,
            "fall_loss_weight": 0.5,
            "keypoint_loss_weight": 0.5,
            "lr": 0.0003,
            "num_workers": 2,
            "occlusion_threshold": 0.3,
            "ojr_epochs": 10,
            "ojr_lr": 0.0005,
            "weight_decay": 1.0e-05
        }
    }
    
    # Write to a new file for Colab
    colab_config_path = "config/colab_config.yaml"
    with open(colab_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created clean config for Colab at: {colab_config_path}")
    print("In Colab, use: python train.py --config config/colab_config.yaml --train-sshfd --gpu 0")

def main():
    """Main function"""
    print("Fixing config.yaml for Google Colab...")
    
    # Fix the existing config
    fix_config_yaml()
    
    # Create a clean config for Colab
    create_colab_config()
    
    print("\nDone! When running in Google Colab:")
    print("1. Make sure to upload both config files to the config directory")
    print("2. Use the command: python train.py --config config/colab_config.yaml --train-sshfd --gpu 0")
    print("3. If you still encounter issues, manually check the config file for YAML syntax errors")

if __name__ == "__main__":
    main()
