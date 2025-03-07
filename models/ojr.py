import torch
import torch.nn as nn


class OJR(nn.Module):
    """
    Occluded Joint Recovery (OJR) model.
    
    This model takes a sequence of keypoints and recovers occluded joints
    using attention mechanisms.
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        """
        Initialize OJR model.
        
        Args:
            input_dim (int): Dimension of input features (num_keypoints * 2)
            hidden_dim (int): Dimension of hidden layers
            num_layers (int): Number of GRU layers
        """
        super(OJR, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU for temporal modeling (non-bidirectional to match checkpoint)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        
        # Attention mechanism (matching the checkpoint)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Prediction network (matching the checkpoint exactly with correct dimensions)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x, occlusion_mask=None):
        """
        Forward pass of OJR model.
        
        Args:
            x (torch.Tensor): Input keypoint sequence [B, T, K*2]
            occlusion_mask (torch.Tensor, optional): Binary mask for occluded joints [K*2]
            
        Returns:
            torch.Tensor: Recovered keypoints [K*2]
        """
        # Get batch and sequence info
        batch_size, seq_len, input_features = x.shape
        
        # Process through GRU
        gru_out, _ = self.gru(x)
        
        # Apply attention
        attention_weights = self.attention(gru_out)
        context = torch.sum(gru_out * attention_weights, dim=1)
        
        # Predict keypoints
        recovered = self.predictor(context)
        
        # Get the original keypoints from the last frame
        original_keypoints = x[:, -1, :]
        
        # If occlusion mask is provided, only replace occluded joints
        if occlusion_mask is not None:
            # Ensure mask has correct dimensionality
            if occlusion_mask.dim() == 1:
                # If 1D, repeat for batch and feature dimensions
                occlusion_mask = occlusion_mask.unsqueeze(0).repeat(batch_size, 1)
            
            # Ensure mask has the same shape as recovered/original keypoints
            if occlusion_mask.shape != original_keypoints.shape:
                # Reshape mask to match the feature dimension
                occlusion_mask = occlusion_mask.view(batch_size, -1)
            
            # Combine original and recovered based on mask
            final_keypoints = torch.where(
                occlusion_mask > 0, 
                recovered, 
                original_keypoints
            )
            
            return final_keypoints
        
        return recovered
    
    def detect_occlusions(self, keypoints, confidence, threshold=0.2):
        """
        Detect occluded keypoints based on confidence scores.
        
        Args:
            keypoints (torch.Tensor): Keypoint coordinates (shape: [num_keypoints*2])
            confidence (torch.Tensor): Confidence scores (shape: [num_keypoints])
            threshold (float): Confidence threshold below which keypoints are considered occluded
            
        Returns:
            torch.Tensor: Binary mask indicating occluded keypoints (1 for occluded, 0 for visible)
        """
        # Ensure proper shape
        if confidence.dim() == 0:  # If it's a scalar tensor
            # Reshape to match expected dimensions
            num_keypoints = len(keypoints) // 2
            confidence = confidence.unsqueeze(0).expand(num_keypoints)
        elif confidence.dim() > 1:  # If it has more than 1 dimension
            confidence = confidence.squeeze()
        
        # Create occlusion mask based on confidence threshold
        occlusion_mask = (confidence < threshold).float()
        
        # Expand mask to match keypoint tensor shape (x,y pairs)
        expanded_mask = torch.zeros_like(keypoints)
        
        # For each keypoint, set both x and y to the same mask value
        num_keypoints = len(keypoints) // 2
        for i in range(num_keypoints):
            expanded_mask[i*2] = occlusion_mask[i]
            expanded_mask[i*2+1] = occlusion_mask[i]
        
        return expanded_mask