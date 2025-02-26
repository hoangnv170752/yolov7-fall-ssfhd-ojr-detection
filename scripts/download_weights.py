#!/usr/bin/env python
"""
Download YOLOv7 weights and clone the repository.

This script:
1. Clones the YOLOv7 repository
2. Downloads pre-trained weights
3. Sets up the project structure
"""

import os
import sys
import argparse
import requests
import subprocess
from pathlib import Path
import shutil
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Download YOLOv7 weights and setup project')
    parser.add_argument('--yolo-dir', type=str, default='models/yolov7', help='Directory to clone YOLOv7 repository')
    parser.add_argument('--weights-dir', type=str, default='weights', help='Directory to save weights')
    parser.add_argument('--no-clone', action='store_true', help='Skip cloning YOLOv7 repository')
    
    return parser.parse_args()


def download_file(url, output_path):
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download
        output_path: Path to save the file
    
    Returns:
        success: True if download was successful, False otherwise
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        
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
    
    except requests.exceptions.RequestException as e:
        print(f"Error during download: {e}")
        return False


def clone_yolov7_repository(repo_url, output_dir):
    """
    Clone YOLOv7 repository.
    
    Args:
        repo_url: Repository URL
        output_dir: Output directory
    
    Returns:
        success: True if cloning was successful, False otherwise
    """
    try:
        print(f"Cloning YOLOv7 repository to {output_dir}...")
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        
        # Check if git is installed
        subprocess.run(['git', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Clone the repository
        subprocess.run(['git', 'clone', repo_url, output_dir], check=True)
        
        print("Repository cloned successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")
        print("Make sure git is installed and accessible in your PATH")
        return False
    except Exception as e:
        print(f"Unexpected error during cloning: {e}")
        return False


def setup_project_structure():
    """
    Set up project directory structure.
    
    This creates the necessary directories if they don't exist.
    """
    directories = [
        'data/train',
        'data/val',
        'data/test',
        'weights',
        'logs',
        'outputs/evaluation',
        'outputs/comparison',
        'docs',
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Project directory structure set up")


def main():
    """Main function."""
    args = parse_args()
    
    # Set repository and weights URLs
    yolov7_repo_url = "https://github.com/WongKinYiu/yolov7.git"
    yolov7_weights_url = "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt"
    
    # Setup project structure
    setup_project_structure()
    
    # Clone YOLOv7 repository if requested
    if not args.no_clone:
        yolo_dir = args.yolo_dir
        
        if os.path.exists(yolo_dir):
            print(f"YOLOv7 directory already exists at {yolo_dir}")
            response = input("Do you want to remove and re-clone? (y/n): ")
            if response.lower() == 'y':
                shutil.rmtree(yolo_dir)
                clone_success = clone_yolov7_repository(yolov7_repo_url, yolo_dir)
                if not clone_success:
                    print("Warning: Failed to clone YOLOv7 repository")
        else:
            clone_success = clone_yolov7_repository(yolov7_repo_url, yolo_dir)
            if not clone_success:
                print("Warning: Failed to clone YOLOv7 repository")
    
    # Download YOLOv7 weights
    weights_path = os.path.join(args.weights_dir, 'yolov7.pt')
    
    if os.path.exists(weights_path):
        print(f"YOLOv7 weights already exist at {weights_path}")
        response = input("Do you want to re-download? (y/n): ")
        if response.lower() == 'y':
            download_success = download_file(yolov7_weights_url, weights_path)
            if not download_success:
                print("Warning: Failed to download YOLOv7 weights")
    else:
        download_success = download_file(yolov7_weights_url, weights_path)
        if not download_success:
            print("Warning: Failed to download YOLOv7 weights")
    
    print("\nSetup complete!")
    print(f"- YOLOv7 repository: {args.yolo_dir}")
    print(f"- YOLOv7 weights: {weights_path}")
    print("\nNext steps:")
    print("1. Organize your dataset")
    print("2. Train the SSHFD model: python train.py --train-sshfd")
    print("3. Train the OJR model: python train.py --train-ojr")
    print("4. Evaluate the models: python evaluate.py")


if __name__ == "__main__":
    main()