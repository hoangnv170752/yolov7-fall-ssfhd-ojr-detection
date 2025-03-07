#!/usr/bin/env python
"""
Organize MultiCam dataset into Fall and NoFall directories.

This script analyzes the MultiCam dataset structure and organizes videos
into Fall and NoFall categories based on the chute number or video content.
"""

import os
import shutil
import argparse
from pathlib import Path
import json
import yaml

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Organize MultiCam dataset into Fall and NoFall directories')
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to the MultiCam dataset')
    parser.add_argument('--output-path', type=str, default='dataset_organized', help='Output directory')
    parser.add_argument('--config', type=str, default='config/fallconfig.yaml', 
                        help='Path to configuration file containing fall classifications')
    parser.add_argument('--create-symlinks', action='store_true', 
                        help='Create symlinks instead of copying files')
    parser.add_argument('--debug', action='store_true', 
                        help='Print detailed debug information')
    
    return parser.parse_args()

def create_folder_structure(output_path):
    """Create the Fall and NoFall directory structure."""
    fall_dir = os.path.join(output_path, 'Fall')
    nofall_dir = os.path.join(output_path, 'NoFall')
    
    os.makedirs(fall_dir, exist_ok=True)
    os.makedirs(nofall_dir, exist_ok=True)
    
    print(f"Created directories:\n  {fall_dir}\n  {nofall_dir}")
    
    return fall_dir, nofall_dir

def load_fall_classification(config_path):
    """
    Load fall classifications from config file.
    
    Expected format in YAML:
    fall_scenarios:
      - chute01
      - chute03
      - ...
    nofall_scenarios:
      - chute02
      - chute04
      - ...
    """
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return {}, {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    fall_scenarios = set(config.get('fall_scenarios', []))
    nofall_scenarios = set(config.get('nofall_scenarios', []))
    
    print(f"Loaded classification from {config_path}:")
    print(f"  Fall scenarios: {len(fall_scenarios)}")
    print(f"  Non-fall scenarios: {len(nofall_scenarios)}")
    
    return fall_scenarios, nofall_scenarios

def create_default_config(dataset_path, config_path):
    """
    Create a default configuration file based on typical fall dataset conventions.
    
    For MultiCam, even-numbered chutes are typically ADLs (no falls) and
    odd-numbered chutes are typically falls, but this varies by dataset version.
    """
    # Get all chute folders
    chute_folders = [d for d in os.listdir(dataset_path) 
                    if os.path.isdir(os.path.join(dataset_path, d)) and d.startswith('chute')]
    
    # For this example, we'll use a simple rule: odd-numbered chutes are falls
    # This is just a heuristic - real classification should be based on dataset documentation
    fall_scenarios = [folder for folder in chute_folders if int(folder.replace('chute', '')) % 2 == 1]
    nofall_scenarios = [folder for folder in chute_folders if int(folder.replace('chute', '')) % 2 == 0]
    
    config = {
        'fall_scenarios': fall_scenarios,
        'nofall_scenarios': nofall_scenarios
    }
    
    # Create the config directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created default configuration at {config_path}")
    print(f"  Fall scenarios: {len(fall_scenarios)}")
    print(f"  Non-fall scenarios: {len(nofall_scenarios)}")
    print("Please review and adjust fall classifications as needed.")
    
    return set(fall_scenarios), set(nofall_scenarios)

def organize_dataset(dataset_path, output_path, fall_scenarios, nofall_scenarios, create_symlinks=False, debug=False):
    """
    Organize the dataset into Fall and NoFall directories.
    
    Args:
        dataset_path: Path to the MultiCam dataset
        output_path: Output directory
        fall_scenarios: Set of scenario folders containing falls
        nofall_scenarios: Set of scenario folders not containing falls
        create_symlinks: Whether to create symlinks instead of copying files
        debug: Print detailed debug information
    """
    fall_dir, nofall_dir = create_folder_structure(output_path)
    
    # Get all directories in the dataset path
    try:
        all_items = os.listdir(dataset_path)
        scenario_folders = [d for d in all_items 
                           if os.path.isdir(os.path.join(dataset_path, d)) and d.startswith('chute')]
        
        if debug:
            print(f"Found {len(scenario_folders)} chute folders in {dataset_path}")
            print(f"All items in directory: {all_items}")
        
        if not scenario_folders:
            print(f"Warning: No chute folders found in {dataset_path}")
            print("Please check the dataset path and structure.")
            return
    except Exception as e:
        print(f"Error accessing dataset directory: {e}")
        return
    
    # Keep track of classified files
    classified_files = {
        'fall': [],
        'nofall': []
    }
    
    total_videos = 0
    
    # Process each scenario folder
    for scenario in scenario_folders:
        scenario_path = os.path.join(dataset_path, scenario)
        
        if debug:
            print(f"\nProcessing scenario: {scenario}")
            print(f"  Path: {scenario_path}")
        
        # Determine if this is a fall scenario
        is_fall = scenario in fall_scenarios
        
        # Handle scenarios not explicitly classified
        if not is_fall and scenario not in nofall_scenarios:
            print(f"Warning: Scenario {scenario} not classified in config. Assuming non-fall.")
            is_fall = False
        
        # Determine destination directory
        dest_dir = fall_dir if is_fall else nofall_dir
        
        if debug:
            print(f"  Classification: {'Fall' if is_fall else 'Non-fall'}")
            print(f"  Destination: {dest_dir}")
        
        # Find all video files in this scenario
        video_files = []
        for root, _, files in os.walk(scenario_path):
            for file in files:
                if file.endswith(('.avi', '.mp4', '.mov')):
                    video_files.append(os.path.join(root, file))
        
        if debug:
            print(f"  Found {len(video_files)} video files")
        
        # Create a directory for this scenario in the destination
        scenario_dest_dir = os.path.join(dest_dir, scenario)
        os.makedirs(scenario_dest_dir, exist_ok=True)
        
        # Copy or link video files
        for video_path in video_files:
            video_name = os.path.basename(video_path)
            dest_path = os.path.join(scenario_dest_dir, video_name)
            
            if create_symlinks:
                # Create a symbolic link
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                try:
                    os.symlink(os.path.abspath(video_path), dest_path)
                    if debug:
                        print(f"  Created symlink: {video_name}")
                except Exception as e:
                    print(f"  Error creating symlink for {video_name}: {e}")
            else:
                # Copy the file
                try:
                    shutil.copy2(video_path, dest_path)
                    if debug:
                        print(f"  Copied: {video_name}")
                except Exception as e:
                    print(f"  Error copying {video_name}: {e}")
            
            # Track the classification
            if is_fall:
                classified_files['fall'].append(dest_path)
            else:
                classified_files['nofall'].append(dest_path)
            
            total_videos += 1
    
    # Create a report of the classification
    report = {
        'fall_count': len(classified_files['fall']),
        'nofall_count': len(classified_files['nofall']),
        'total_videos': total_videos,
        'fall_scenarios': list(sorted(fall_scenarios)),
        'nofall_scenarios': list(sorted(nofall_scenarios))
    }
    
    report_path = os.path.join(output_path, 'classification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\nOrganization completed!")
    print(f"Total videos processed: {total_videos}")
    print(f"Fall videos: {report['fall_count']}")
    print(f"Non-fall videos: {report['nofall_count']}")
    print(f"Report saved to: {report_path}")

def main():
    """Main function."""
    args = parse_args()
    
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path does not exist: {args.dataset_path}")
        return
    
    # Load fall classifications from config
    fall_scenarios, nofall_scenarios = load_fall_classification(args.config)
    
    # If classifications are not available, create a default config
    if not fall_scenarios and not nofall_scenarios:
        try:
            fall_scenarios, nofall_scenarios = create_default_config(args.dataset_path, args.config)
        except Exception as e:
            print(f"Error creating default configuration: {e}")
            print("Please specify a valid configuration file.")
            return
    
    # Organize the dataset
    organize_dataset(
        args.dataset_path, 
        args.output_path, 
        fall_scenarios, 
        nofall_scenarios,
        args.create_symlinks,
        args.debug
    )

if __name__ == "__main__":
    main()