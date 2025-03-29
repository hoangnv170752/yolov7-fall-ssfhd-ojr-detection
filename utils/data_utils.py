import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.transforms as transforms
from pathlib import Path
import logging
import random
from .augmentation import VideoAugmenter, OverSampler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataset_errors.log"),
        logging.StreamHandler()
    ]
)


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
        self.video_paths = []
        self.labels = []
        
        with open(video_list, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split(',')
                    if len(parts) == 2:
                        self.video_paths.append(parts[0])
                        self.labels.append(int(parts[1]))
        
        # Initialize augmenter for training mode
        self.augmenter = VideoAugmenter(p=0.5) if mode == 'train' else None
        
        # Log dataset statistics
        class_counts = {}
        for label in self.labels:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1
        
        logging.info(f"Loaded {len(self.video_paths)} videos for {mode} set")
        logging.info(f"Class distribution: {class_counts}")
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            frames: Tensor of shape (temporal_window, 3, H, W)
            label: Tensor of shape (1,)
        """
        video_path = self.data_root / self.video_paths[idx]
        label = self.labels[idx]
        
        try:
            frames = self._extract_frames(video_path)
            
            if frames is None:
                # If frame extraction fails, return a dummy tensor
                logging.warning(f"Failed to extract frames from {video_path}, using dummy frames")
                frames = torch.zeros((self.temporal_window, 3, 224, 224))
            
            # Apply augmentation for training mode
            if self.augmenter is not None and self.mode == 'train':
                frames = self.augmenter(frames)
            
            # Apply additional transforms if provided
            if self.transform:
                frames = self.transform(frames)
            
            # Return frames and label
            if self.mode == 'train' or self.mode == 'val' or (self.mode == 'test' and label is not None):
                label = torch.tensor(label, dtype=torch.long)
                return frames, label
            else:
                return frames
            
        except Exception as e:
            logging.error(f"Error processing {video_path}: {str(e)}")
            # Return dummy data in case of error
            frames = torch.zeros((self.temporal_window, 3, 224, 224))
            return frames, label
    
    def _extract_frames(self, video_path):
        """
        Extract frames from video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            torch.Tensor: Tensor of frames with shape (T, C, H, W)
        """
        # Try different backends for video capture
        cap = None
        backends = [
            cv2.CAP_FFMPEG,       # Try FFMPEG first (most reliable)
            1200,                 # Try backend 1200 (seen in logs)
            1900,                 # Try backend 1900 (seen in logs)
            cv2.CAP_AVFOUNDATION, # Try AVFoundation (for macOS)
            cv2.CAP_ANY           # Try any available backend
        ]
        
        for backend_id in backends:
            try:
                logging.info(f"Trying to open {video_path} with backend {backend_id}")
                cap = cv2.VideoCapture(str(video_path), backend_id)
                if cap.isOpened():
                    # Verify we can read a frame
                    ret, test_frame = cap.read()
                    if ret:
                        # Reset to beginning
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        logging.info(f"Successfully opened {video_path} with backend {backend_id}")
                        break
                    else:
                        cap.release()
                        cap = None
            except Exception as e:
                logging.warning(f"Error with backend {backend_id} for {video_path}: {str(e)}")
                if cap is not None:
                    cap.release()
                    cap = None
        
        # If all backends failed, try without specifying a backend
        if cap is None or not cap.isOpened():
            try:
                logging.info(f"Trying to open {video_path} without specifying backend")
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened() or not cap.read()[0]:
                    logging.error(f"Failed to open {video_path} with any backend")
                    return self._generate_dummy_frames()
            except Exception as e:
                logging.error(f"Error opening {video_path}: {str(e)}")
                return self._generate_dummy_frames()
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count <= 0:
            logging.error(f"Video has no frames: {video_path}")
            cap.release()
            return self._generate_dummy_frames()
        
        # Determine frames to extract
        if frame_count <= self.temporal_window:
            # If video has fewer frames than temporal window, use all frames
            frame_indices = list(range(frame_count))
            # Pad with the last frame if needed
            if len(frame_indices) < self.temporal_window:
                frame_indices.extend([frame_count-1] * (self.temporal_window - len(frame_indices)))
        else:
            # Sample frames evenly across the video
            frame_indices = np.linspace(0, frame_count - 1, self.temporal_window, dtype=int)
        
        # Extract frames
        frames = []
        for idx in frame_indices:
            success = cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            if not success:
                logging.warning(f"Failed to set frame position to {idx} for {video_path}")
                
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"Failed to read frame {idx} from {video_path}")
                # Use a black frame as placeholder
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame
            frame = cv2.resize(frame, (224, 224))
            
            # Convert to float and normalize
            frame = frame.astype(np.float32) / 255.0
            
            # Convert to tensor [C, H, W]
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1)
            frames.append(frame_tensor)
        
        # Release video capture
        cap.release()
        
        # Check if we got any frames
        if len(frames) == 0:
            logging.error(f"No frames extracted from {video_path}")
            return self._generate_dummy_frames()
            
        # Ensure we have the correct number of frames
        if len(frames) != self.temporal_window:
            logging.warning(f"Expected {self.temporal_window} frames but got {len(frames)} from {video_path}")
            # Pad with the last frame or truncate
            if len(frames) < self.temporal_window:
                last_frame = frames[-1]
                frames.extend([last_frame] * (self.temporal_window - len(frames)))
            else:
                frames = frames[:self.temporal_window]
            
        # Convert list of tensors to a single tensor
        frames_tensor = torch.stack(frames, dim=0)
        return frames_tensor
        
    def _generate_dummy_frames(self):
        """Generate dummy frames when video loading fails"""
        # Create a tensor of zeros with the correct shape
        dummy_frames = torch.zeros((self.temporal_window, 3, 224, 224))
        return dummy_frames
    
    def _adjust_frames_count(self, frames):
        """Adjust the number of frames to match temporal_window"""
        if len(frames) > self.temporal_window:
            # If we have too many frames, truncate
            return frames[:self.temporal_window]
        elif len(frames) < self.temporal_window:
            # If we have too few frames, duplicate the last frame
            last_frame = frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8)
            while len(frames) < self.temporal_window:
                frames.append(last_frame.copy())
        return frames


def custom_collate_fn(batch):
    """
    Custom collate function for DataLoader.
    
    Args:
        batch: List of tuples (frames, label)
        
    Returns:
        frames_batch: Tensor of shape (B, T, C, H, W)
        labels_batch: Tensor of shape (B,)
    """
    # Filter out None values
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    # Separate frames and labels
    frames, labels = zip(*batch)
    
    # Stack frames along batch dimension
    frames_batch = torch.stack(frames, dim=0)
    
    # Stack labels along batch dimension
    labels_batch = torch.stack(labels, dim=0)
    
    return frames_batch, labels_batch


def create_dataloaders(config):
    """
    Create dataloaders for training, validation, and testing.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        tuple: Train, validation, and test dataloaders
    """
    data_root = config['data']['root']
    train_list = config['data']['train_list']
    val_list = config['data']['val_list']
    test_list = config['data']['test_list']
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']
    temporal_window = config['model']['temporal_window']
    
    # Create datasets
    train_dataset = FallDetectionDataset(
        data_root=data_root,
        video_list=train_list,
        temporal_window=temporal_window,
        mode='train'
    )
    
    val_dataset = FallDetectionDataset(
        data_root=data_root,
        video_list=val_list,
        temporal_window=temporal_window,
        mode='val'
    )
    
    test_dataset = FallDetectionDataset(
        data_root=data_root,
        video_list=test_list,
        temporal_window=temporal_window,
        mode='test'
    )
    
    # Apply oversampling to the training dataset
    try:
        oversampler = OverSampler(train_dataset, minority_class=1, oversample_ratio=1.5)
        oversampled_indices = oversampler.get_oversampled_indices()
        
        # Create a sampler with the oversampled indices
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(oversampled_indices)
    except Exception as e:
        logging.error(f"Error in oversampling: {str(e)}, using random sampler instead")
        train_sampler = None
    
    # Create dataloaders with error handling
    try:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler if train_sampler is not None else None,
            shuffle=train_sampler is None,  # Only shuffle if not using sampler
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn,
            drop_last=False
        )
        
        return train_loader, val_loader, test_loader
    
    except Exception as e:
        logging.error(f"Error creating dataloaders: {str(e)}")
        # Fallback to simpler configuration
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=custom_collate_fn,
            drop_last=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=custom_collate_fn,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=custom_collate_fn,
            drop_last=False
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