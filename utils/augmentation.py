"""
Data augmentation utilities for fall detection training.

This module provides data augmentation techniques to improve
model generalization and address class imbalance.
"""

import torch
import numpy as np
import random
import cv2
import logging


class VideoAugmenter:
    """Class for applying augmentations to video frames."""
    
    def __init__(self, p=0.5):
        """
        Initialize VideoAugmenter.
        
        Args:
            p (float): Probability of applying each augmentation
        """
        self.p = p
        
    def __call__(self, frames):
        """
        Apply augmentations to frames.
        
        Args:
            frames (torch.Tensor): Video frames tensor of shape [T, C, H, W]
            
        Returns:
            torch.Tensor: Augmented frames
        """
        try:
            # Ensure frames is a tensor
            if not isinstance(frames, torch.Tensor):
                logging.warning("Frames is not a tensor, skipping augmentation")
                return frames
                
            # Check tensor shape
            if len(frames.shape) != 4:
                logging.warning(f"Expected 4D tensor [T,C,H,W], got shape {frames.shape}, skipping augmentation")
                return frames
                
            # Apply random horizontal flip
            if random.random() < self.p:
                frames = self._horizontal_flip(frames)
            
            # Apply random brightness adjustment
            if random.random() < self.p:
                frames = self._random_brightness(frames)
            
            # Apply random contrast adjustment
            if random.random() < self.p:
                frames = self._random_contrast(frames)
                
            return frames
            
        except Exception as e:
            logging.error(f"Error in augmentation: {str(e)}")
            return frames  # Return original frames if augmentation fails
    
    def _horizontal_flip(self, frames):
        """Horizontally flip the frames."""
        return frames.flip(dims=[-1])
    
    def _random_brightness(self, frames):
        """Apply random brightness adjustment."""
        # Random brightness factor between 0.8 and 1.2
        factor = random.uniform(0.8, 1.2)
        return torch.clamp(frames * factor, 0, 1)
    
    def _random_contrast(self, frames):
        """Apply random contrast adjustment."""
        # Random contrast factor between 0.8 and 1.2
        factor = random.uniform(0.8, 1.2)
        mean = frames.mean(dim=[-1, -2], keepdim=True)
        return torch.clamp((frames - mean) * factor + mean, 0, 1)


class OverSampler:
    """Class for oversampling minority class samples."""
    
    def __init__(self, dataset, minority_class=1, oversample_ratio=1.5):
        """
        Initialize OverSampler.
        
        Args:
            dataset: Dataset to oversample
            minority_class (int): Class index to oversample
            oversample_ratio (float): Ratio of minority to majority samples after oversampling
        """
        self.dataset = dataset
        self.minority_class = minority_class
        self.oversample_ratio = oversample_ratio
        
    def get_oversampled_indices(self):
        """
        Get indices for oversampled dataset.
        
        Returns:
            list: Indices to sample from the original dataset
        """
        try:
            # Count samples per class
            class_counts = {}
            for i in range(len(self.dataset)):
                try:
                    _, label = self.dataset[i]
                    label_val = label.item() if isinstance(label, torch.Tensor) else label
                    if label_val not in class_counts:
                        class_counts[label_val] = 0
                    class_counts[label_val] += 1
                except Exception as e:
                    logging.error(f"Error accessing dataset item {i}: {str(e)}")
                    continue
            
            # Determine majority class count
            if not class_counts:
                logging.warning("No valid class counts found, returning original indices")
                return list(range(len(self.dataset)))
                
            majority_count = max(class_counts.values())
            minority_count = class_counts.get(self.minority_class, 0)
            
            logging.info(f"Class distribution before oversampling: {class_counts}")
            
            # Calculate target count for minority class
            target_count = int(majority_count * self.oversample_ratio)
            
            # Get indices of minority class samples
            minority_indices = []
            for i in range(len(self.dataset)):
                try:
                    _, label = self.dataset[i]
                    label_val = label.item() if isinstance(label, torch.Tensor) else label
                    if label_val == self.minority_class:
                        minority_indices.append(i)
                except Exception as e:
                    continue
            
            # Create oversampled indices
            all_indices = list(range(len(self.dataset)))
            
            # Add additional minority samples if needed
            if minority_count < target_count and minority_count > 0:
                # Number of times to repeat each minority sample on average
                repeat_factor = target_count / minority_count
                
                # Add repeated minority samples
                additional_indices = []
                for _ in range(int(repeat_factor) - 1):
                    additional_indices.extend(minority_indices)
                
                # Add remaining samples randomly
                remaining = target_count - (minority_count * int(repeat_factor))
                if remaining > 0 and minority_indices:
                    additional_indices.extend(random.choices(minority_indices, k=int(remaining)))
                
                all_indices.extend(additional_indices)
                
                logging.info(f"Added {len(additional_indices)} oversampled minority class examples")
            
            # Shuffle indices
            random.shuffle(all_indices)
            
            return all_indices
            
        except Exception as e:
            logging.error(f"Error in oversampling: {str(e)}")
            return list(range(len(self.dataset)))


def apply_mixup(frames1, labels1, frames2, labels2, alpha=0.2):
    """
    Apply mixup augmentation to a pair of samples.
    
    Args:
        frames1 (torch.Tensor): First batch of frames
        labels1 (torch.Tensor): Labels for first batch
        frames2 (torch.Tensor): Second batch of frames
        labels2 (torch.Tensor): Labels for second batch
        alpha (float): Mixup interpolation strength
        
    Returns:
        tuple: Mixed frames and labels
    """
    # Generate mixup coefficient
    lam = np.random.beta(alpha, alpha)
    
    # Mix frames
    mixed_frames = lam * frames1 + (1 - lam) * frames2
    
    # For classification, we return both labels and the lambda
    return mixed_frames, labels1, labels2, lam
