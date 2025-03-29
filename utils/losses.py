"""
Custom loss functions for fall detection model training.

This module contains implementations of various loss functions
specifically designed for training fall detection models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for classification tasks.
    
    Focal Loss is designed to address class imbalance by down-weighting easy examples
    and focusing training on hard negatives.
    
    Reference: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, weight=None, reduction='mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha (float): Weighting factor for the rare class (default: 1.0)
            gamma (float): Focusing parameter that adjusts the rate at which easy examples
                          are down-weighted (default: 2.0)
            weight (torch.Tensor): Manual rescaling weight for each class (default: None)
            reduction (str): Specifies the reduction to apply to the output:
                            'none' | 'mean' | 'sum' (default: 'mean')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Forward pass.
        
        Args:
            inputs (torch.Tensor): Model predictions (logits)
            targets (torch.Tensor): Ground truth labels
            
        Returns:
            torch.Tensor: Computed loss
        """
        # Apply softmax to get probabilities
        logpt = F.log_softmax(inputs, dim=1)
        
        # Gather the log probabilities for the target classes
        logpt = logpt.gather(1, targets.unsqueeze(1))
        logpt = logpt.squeeze(1)
        
        # Convert to probabilities
        pt = torch.exp(logpt)
        
        # Apply class weights if provided
        if self.weight is not None:
            # Gather the weights for the target classes
            at = self.weight.gather(0, targets)
            logpt = logpt * at
        
        # Compute focal loss
        focal_weight = (1 - pt) ** self.gamma
        loss = -self.alpha * focal_weight * logpt
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
