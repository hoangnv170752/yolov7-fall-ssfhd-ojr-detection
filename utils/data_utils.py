import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path


class FallDetectionDataset(Dataset):
    """
    Dataset for fall detection from videos.
    
    This dataset loads videos from disk, extracts frames at regular intervals,
    and applies transformations to prepare them for the model.
    """
    
    def __init__(self, data_root, video_list, transform=None, temporal_window=5, mode='train'):
        """
        Initialize the dataset.
        
        Args:
            data_root: Root directory containing videos
            video_list: Path to text file with video paths and labels
            transform: Torchvision transforms to apply to frames
            temporal_window: Number of frames to extract from each video
            mode: 'train', 'val', or 'test'
        """
        self.data_root = Path(data_root)
        self.transform = transform
        self.temporal_window = temporal_window
        self.mode = mode
        
        # Load video paths and labels
        with open(video_list, 'r') as f:
            self.videos = [line.strip().split(',') for line in f.readlines()]
            
        if self.mode == 'train' or self.mode == 'val':
            # Extract video paths and labels (0: no fall, 1: fall)
            self.video_paths = [video[0] for video in self.videos]
            self.labels = [int(video[1]) for video in self.videos]
        else:
            # For test mode, we might not have labels
            self.video_paths = [video[0] for video in self.videos]
            self.labels = [int(video[1]) if len(video) > 1 else None for video in self.videos]
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.data_root / self.video_paths[idx]
        
        # Extract frames from video
        frames = self._extract_frames(video_path)
        
        # Apply transformations
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        
        # Stack frames
        frames = torch.stack(frames, dim=0)
        
        # Return frames and label
        if self.mode == 'train' or self.mode == 'val' or (self.mode == 'test' and self.labels[idx] is not None):
            label = self.labels[idx]
            return frames, label
        else:
            return frames
    
    def _extract_frames(self, video_path):
        """
        Extract frames from video at regular intervals.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            frames: List of extracted frames
        """
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count <= self.temporal_window:
            sample_indices = list(range(frame_count))
            sample_indices += [frame_count-1] * (self.temporal_window - frame_count)
        else:
            # Take evenly spaced frames
            sample_indices = np.linspace(0, frame_count-1, self.temporal_window, dtype=int)
        
        frames = []
        for i in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        return frames


def create_dataloaders(config):
    """
    Create dataloaders for training, validation, and testing.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        train_loader, val_loader, test_loader: DataLoader objects
    """
    data_root = config['data']['root']
    batch_size = config['training']['batch_size']
    
    # Define transformations
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = FallDetectionDataset(
        data_root=data_root,
        video_list=config['data']['train_list'],
        transform=train_transform,
        temporal_window=config['model']['temporal_window'],
        mode='train'
    )
    
    val_dataset = FallDetectionDataset(
        data_root=data_root,
        video_list=config['data']['val_list'],
        transform=val_transform,
        temporal_window=config['model']['temporal_window'],
        mode='val'
    )
    
    test_dataset = FallDetectionDataset(
        data_root=data_root,
        video_list=config['data']['test_list'],
        transform=val_transform,
        temporal_window=config['model']['temporal_window'],
        mode='test'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


class KeypointDataset(Dataset):
    """
    Dataset for training the OJR model with keypoint data.
    
    This dataset uses pre-extracted keypoints rather than raw video
    to accelerate OJR training.
    """
    
    def __init__(self, keypoint_data_path, temporal_window=5, create_occlusions=True, occlusion_rate=0.3):
        """
        Initialize the dataset.
        
        Args:
            keypoint_data_path: Path to saved keypoint data
            temporal_window: Number of frames in sequence
            create_occlusions: Whether to create synthetic occlusions
            occlusion_rate: Rate of synthetic occlusions
        """
        self.keypoint_data = torch.load(keypoint_data_path)
        self.temporal_window = temporal_window
        self.create_occlusions = create_occlusions
        self.occlusion_rate = occlusion_rate
    
    def __len__(self):
        return len(self.keypoint_data)
    
    def __getitem__(self, idx):
        # Get keypoint sequence
        keypoint_seq = self.keypoint_data[idx]['keypoints']
        confidence = self.keypoint_data[idx]['confidence']
        
        # Create synthetic occlusions
        if self.create_occlusions:
            occlusion_mask = self._create_random_occlusions(keypoint_seq[-1], confidence[-1])
        else:
            occlusion_mask = torch.zeros_like(keypoint_seq[-1])
        
        return {
            'keypoint_seq': keypoint_seq,
            'confidence': confidence,
            'occlusion_mask': occlusion_mask,
            'target': keypoint_seq[-1]
        }
    
    def _create_random_occlusions(self, keypoints, confidence):
        """Create random occlusions for training."""
        num_keypoints = confidence.size(0)
        
        # Random binary mask (1 = occluded)
        mask = torch.rand(num_keypoints) < self.occlusion_rate
        
        # Expand mask for x,y coordinates
        expanded_mask = torch.zeros_like(keypoints)
        expanded_mask[0::2] = mask
        expanded_mask[1::2] = mask 
        
        return expanded_mask