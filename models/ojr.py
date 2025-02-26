import torch
import torch.nn as nn
import torch.nn.functional as F


class OJR(nn.Module):
    """
    Occluded Joints Recovery (OJR) model.
    
    This model uses temporal information to recover occluded keypoints
    by analyzing the sequence of previous frames and predicting the
    locations of joints that are currently not visible.
    """
    
    def __init__(self, input_dim=34, hidden_dim=128, num_layers=2):
        """
        Initialize OJR model.
        
        Args:
            input_dim: Dimension of input features (num_keypoints * 2 for x,y coordinates)
            hidden_dim: Dimension of hidden features
            num_layers: Number of GRU layers
        """
        super(OJR, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU to model temporal dependencies
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, keypoint_sequence, occlusion_mask=None):
        """
        Forward pass.
        
        Args:
            keypoint_sequence: Tensor of shape (batch_size, seq_len, num_keypoints*2)
                              containing keypoint coordinates
            occlusion_mask: Binary mask of occluded keypoints (batch_size, num_keypoints*2)
                           where 1 indicates occluded, 0 indicates visible
                           
        Returns:
            recovered_keypoints: Keypoints with occluded joints recovered
        """
        batch_size = keypoint_sequence.size(0)
        seq_len = keypoint_sequence.size(1)
        
        # Handle missing occlusion mask
        if occlusion_mask is None:
            # If no occlusion mask provided, assume no occlusions
            occlusion_mask = torch.zeros(batch_size, keypoint_sequence.size(2)).to(keypoint_sequence.device)
        
        # Run GRU on sequence
        all_hidden, _ = self.gru(keypoint_sequence)
        
        attention_scores = []
        for t in range(seq_len):
            score = self.attention(all_hidden[:, t, :])
            attention_scores.append(score)
        
        attention_scores = torch.cat(attention_scores, dim=1) 
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(2) 
        
        # Apply attention weights
        context = torch.sum(all_hidden * attention_weights, dim=1)
        
        # Predict keypoints
        predicted_keypoints = self.predictor(context)
        # Get the last frame's keypoints as the base
        last_frame_keypoints = keypoint_sequence[:, -1, :]
        
        # Replace occluded keypoints with predictions
        recovered_keypoints = last_frame_keypoints.clone()
        recovered_keypoints[occlusion_mask > 0] = predicted_keypoints[occlusion_mask > 0]
        
        return recovered_keypoints
    
    def detect_occlusions(self, keypoints, confidence, threshold=0.3):
        """
        Detect occluded joints based on confidence scores.
        
        Args:
            keypoints: Keypoint coordinates (batch_size, num_keypoints*2)
            confidence: Confidence scores (batch_size, num_keypoints)
            threshold: Confidence threshold below which keypoints are considered occluded
            
        Returns:
            occlusion_mask: Binary mask of occluded joints (batch_size, num_keypoints*2)
        """
        batch_size = keypoints.size(0)
        num_keypoints = confidence.size(1)
        
        # Create mask from confidence scores
        mask = (confidence < threshold).float()
        
        # Expand mask for x,y coordinates
        occlusion_mask = torch.zeros_like(keypoints)
        occlusion_mask[:, 0::2] = mask
        occlusion_mask[:, 1::2] = mask
        
        return occlusion_mask