#!/usr/bin/env python
"""
Generate list files for MP4 videos.

This script creates train_list_mp4.txt, val_list_mp4.txt, and test_list_mp4.txt
by converting the paths in the original list files to point to MP4 files.
"""

import os
import glob
from pathlib import Path

def generate_mp4_lists():
    """Generate list files for MP4 videos."""
    data_dir = "data"
    
    # Create mappings for original list files
    list_files = {
        "train": os.path.join(data_dir, "train_list.txt"),
        "val": os.path.join(data_dir, "val_list.txt"),
        "test": os.path.join(data_dir, "test_list.txt")
    }
    
    # Create mappings for new MP4 list files
    mp4_list_files = {
        "train": os.path.join(data_dir, "train_list_mp4.txt"),
        "val": os.path.join(data_dir, "val_list_mp4.txt"),
        "test": os.path.join(data_dir, "test_list_mp4.txt")
    }
    
    # Process each list file
    for split, list_file in list_files.items():
        mp4_entries = []
        
        # Check if the original list file exists
        if not os.path.exists(list_file):
            print(f"Warning: {list_file} does not exist. Creating empty MP4 list file.")
            with open(mp4_list_files[split], 'w') as f:
                pass
            continue
        
        # Read the original list file
        with open(list_file, 'r') as f:
            entries = f.readlines()
        
        # Process each entry
        for entry in entries:
            parts = entry.strip().split(',')
            if len(parts) < 2:
                continue
                
            path, label = parts[0], parts[1]
            
            # Convert path to MP4
            original_path = Path(path)
            filename = original_path.stem
            directory = original_path.parent
            
            # Create new path with MP4 extension
            mp4_path = f"{directory}_mp4/{filename}.mp4"
            
            # Check if the MP4 file exists
            if os.path.exists(os.path.join(data_dir, mp4_path)):
                mp4_entries.append(f"{mp4_path},{label}\n")
            else:
                print(f"Warning: MP4 file not found for {path}")
        
        # Write the new MP4 list file
        with open(mp4_list_files[split], 'w') as f:
            f.writelines(mp4_entries)
        
        print(f"Created {mp4_list_files[split]} with {len(mp4_entries)} entries")

    # If no entries were found in the original list files, create entries based on MP4 files in the directories
    for split in ["train", "val", "test"]:
        mp4_list_file = mp4_list_files[split]
        
        # Check if the file is empty
        if os.path.exists(mp4_list_file) and os.path.getsize(mp4_list_file) == 0:
            print(f"No entries found in original list file for {split}. Creating entries based on MP4 files.")
            
            # Find all MP4 files in the directory
            mp4_dir = os.path.join(data_dir, f"{split}_mp4")
            if not os.path.exists(mp4_dir):
                print(f"Warning: {mp4_dir} does not exist.")
                continue
                
            mp4_files = glob.glob(os.path.join(mp4_dir, "*.mp4"))
            
            # Create entries for each MP4 file
            mp4_entries = []
            for mp4_file in mp4_files:
                rel_path = os.path.relpath(mp4_file, data_dir)
                
                # Determine label based on filename
                filename = os.path.basename(mp4_file)
                label = 1 if "fall" in filename.lower() else 0
                
                mp4_entries.append(f"{rel_path},{label}\n")
            
            # Write the new MP4 list file
            with open(mp4_list_file, 'w') as f:
                f.writelines(mp4_entries)
            
            print(f"Created {mp4_list_file} with {len(mp4_entries)} entries based on MP4 files")

if __name__ == "__main__":
    generate_mp4_lists()
