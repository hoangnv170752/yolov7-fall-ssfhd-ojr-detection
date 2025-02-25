#!/usr/bin/env python
"""
Download YOLOv7 weights and clone the repository.

This script:
1. Clones the YOLOv7 repository
2. Downloads pre-trained weights
3. Sets up the project structure
"""

import os
import argparse
import requests
import subprocess
from pathlib import Path
import yaml
import sys
import shutil
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Download YOLOv7 weights and setup project')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--no-clone', action='store_true', help='Skip cloning YOLOv7 repository')
    parser.add_argument('--weights-only', action='store_true', help='Download weights only')
    
    return parser.parse_args()


def download_file(url, output_path):
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download
        output_path: Path to save the file
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    progress_bar = tqdm(
        total=total_size,
        unit='B',
        unit_scale=True,
        desc=f"Downloading {os.path.basename(output_path)}"
    )
    
    with open(output_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    
    progress_bar.close()
    
    if total_size != 0 and progress_bar.n != total_size:
        print("ERROR: Download incomplete")
        return False
    
    return True


def clone_yolov7(repo_url, output_dir):
    """
    Clone YOLOv7 repository.
    
    Args:
        repo_url: Repository URL
        output_dir: Output directory
    """
    try:
        print(f"Cloning YOLOv7 repository to {output_dir}...")
        subprocess.run(['git', 'clone', repo_url, output_dir], check=True)
        print("Repository cloned successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")
        return False


def download_weights(weights_url, output_path):
    """
    Download YOLOv7 weights.
    
    Args:
        weights_url: URL for weights
        output_path: Path to save weights
    """
    print(f"Downloading YOLOv7 weights to {output_path}...")
    success = download_file(weights_url, output_path)
    
    if success:
        print("Weights downloaded successfully")
    else:
        print("Failed to download weights")
    
    return success


def setup_project_structure(project_root):
    """
    Set up project directory structure.
    
    Args:
        project_root: Project root directory
    """
    directories = [
        'data/train',
        'data/val',
        'data/test',
        'weights',
        'logs',
        'outputs'
    ]
    
    for directory in directories:
        os.makedirs(os.path.join(project_root, directory), exist_ok=True)
    
    print("Project directory structure set up")


def main():
    """Main function."""
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Project setup
    project_root = Path(__file__).resolve().parents[1]
    
    # URLs
    repo_url = "https://github.com/WongKinYiu/yolov7.git"
    weights_url = "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt"
    
    # Clone YOLOv7 repository
    yolo_path = os.path.join(project_root, 'models', 'yolov7')
    
    if not args.no_clone and not args.weights_only:
        if os.path.exists(yolo_path):
            print(f"YOLOv7 directory already exists at {yolo_path}")
            response = input("Do you want to remove and re-clone? (y/n): ")
            if response.lower() == 'y':
                shutil.rmtree(yolo_path)
                clone_success = clone_yolov7(repo_url, yolo_path)
                if not clone_success:
                    sys.exit(1)
        else:
            clone_success = clone_yolov7(repo_url, yolo_path)
            if not clone_success:
                sys.exit(1)
    
    # Download weights
    weights_path = os.path.join(project_root, config['model']['yolo_weights'])
    
    if os.path.exists(weights_path):
        print(f"Weights file already exists at {weights_path}")
        response = input("Do you want to re-download? (y/n): ")
        if response.lower() == 'y':
            download_success = download_weights(weights_url, weights_path)
            if not download_success:
                print("Warning: Failed to download weights")
    else:
        download_success = download_weights(weights_url, weights_path)
        if not download_success:
            print("Warning: Failed to download weights")
    
    # Set up project structure
    if not args.weights_only:
        setup_project_structure(project_root)
    
    print("Setup complete!")


if __name__ == "__main__":
    main()