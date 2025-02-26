import torch
import torch.nn as nn
import torch.nn.functional as F


class SSHFD(nn.Module):
    """
    Smart System for Human Fall Detection (SSHFD).
    
    This model combines keypoint estimation and fall detection capabilities.
    It processes input frames to estimate human keypoints and analyze temporal
    patterns to detect falling behavior.
    """
    
    def __init__(self, num_keypoints=17, hidden_dim=256, temporal_window=5):
        """
        Initialize SSHFD model.
        
        Args:
            num_keypoints: Number of keypoints to detect (default: 17 for COCO format)
            hidden_dim: Dimension of hidden features
            temporal_window: Number of frames to consider for temporal analysis
        """
        super(SSHFD, self).__init__()
        self.num_keypoints = num_keypoints
        self.temporal_window = temporal_window
        
        # Keypoint estimation network
        self.keypoint_branch = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_keypoints, kernel_size=1, stride=1)
        )
        
        # Temporal feature extractor
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(num_keypoints*2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Fall classification branch
        self.fall_classifier = nn.Sequential(
            nn.Linear(hidden_dim * (temporal_window//2), 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2) 
        )
    
    def forward(self, x, bboxes=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, temporal_window, 3, H, W)
               or (batch_size, 3, H, W) for single frame
            bboxes: Person bounding boxes (batch_size, 4) with format (x1, y1, x2, y2)
                   If None, use full frame
                   
        Returns:
            keypoints: Estimated keypoints
            fall_prediction: Fall classification scores (None for single frame)
        """
        batch_size = x.size(0)
        
        # Handle single frame input
        if len(x.shape) == 4:
            single_frame = True
            # Add temporal dimension
            x = x.unsqueeze(1)
        else:
            single_frame = False
            
        temporal_dim = x.size(1)
        
        # Process each frame to extract keypoints
        all_keypoints = []
        all_confidences = []
        
        for t in range(temporal_dim):
            frame_features = self.keypoint_branch(x[:, t])
            
            # If bounding boxes provided, extract keypoints from ROI
            if bboxes is not None:
                keypoints = []
                confidences = []
                for i, bbox in enumerate(bboxes):
                    x1, y1, x2, y2 = bbox
                    roi_features = frame_features[i, :, int(y1):int(y2), int(x1):int(x2)]
                    # Use spatial softmax to get keypoint locations
                    kpts, conf = self._extract_keypoints_softmax(roi_features)
                    keypoints.append(kpts)
                    confidences.append(conf)
                keypoints = torch.stack(keypoints, dim=0)
                confidences = torch.stack(confidences, dim=0)
            else:
                # Extract keypoints from full frame
                keypoints, confidences = self._extract_keypoints_softmax(frame_features)
                
            all_keypoints.append(keypoints)
            all_confidences.append(confidences)
            
        # Stack keypoints along temporal dimension
        keypoints_sequence = torch.stack(all_keypoints, dim=1)
        confidences_sequence = torch.stack(all_confidences, dim=1)
        
        # If single frame, no fall detection
        if single_frame:
            return keypoints_sequence.squeeze(1), confidences_sequence.squeeze(1), None
            
        # Prepare for temporal convolution - reshape to (batch_size, channels, temporal_dim)
        keypoints_temporal = keypoints_sequence.reshape(batch_size, self.num_keypoints*2, temporal_dim)
        
        # Extract temporal features
        temporal_features = self.temporal_conv(keypoints_temporal)
        
        # Classify fall
        temporal_features_flat = temporal_features.reshape(batch_size, -1)
        fall_prediction = self.fall_classifier(temporal_features_flat)
        
        return keypoints_sequence, confidences_sequence, fall_prediction
    
    def _extract_keypoints_softmax(self, feature_maps):
        """
        Extract keypoint coordinates using spatial softmax.
        
        Args:
            feature_maps: Feature maps of shape (batch_size, num_keypoints, height, width)
            
        Returns:
            keypoints: Normalized keypoint coordinates of shape (batch_size, num_keypoints*2)
            confidences: Confidence scores for each keypoint of shape (batch_size, num_keypoints)
        """
        batch_size = feature_maps.size(0)
        num_keypoints = feature_maps.size(1)
        height = feature_maps.size(2)
        width = feature_maps.size(3)
        
        # Reshape for spatial softmax
        feature_maps_flat = feature_maps.reshape(batch_size, num_keypoints, -1)
        
        # Get max values for confidence
        max_values, _ = torch.max(feature_maps_flat, dim=2)
        confidences = torch.sigmoid(max_values)
        
        # Apply softmax to get probability distribution
        feature_maps_softmax = F.softmax(feature_maps_flat, dim=2)
        feature_maps_softmax = feature_maps_softmax.reshape(batch_size, num_keypoints, height, width)
        
        # Generate coordinate maps
        x_coords = torch.arange(width).float().to(feature_maps.device)
        y_coords = torch.arange(height).float().to(feature_maps.device)
        
        x_coords = x_coords.view(1, 1, 1, width).expand(batch_size, num_keypoints, height, width)
        y_coords = y_coords.view(1, 1, height, 1).expand(batch_size, num_keypoints, height, width)
        
        # Calculate expected coordinates (weighted average)
        x_mean = torch.sum(feature_maps_softmax * x_coords, dim=(2, 3))
        y_mean = torch.sum(feature_maps_softmax * y_coords, dim=(2, 3))
        
        # Normalize to [0, 1]
        x_mean = x_mean / width
        y_mean = y_mean / height
        
        # Interleave x and y coordinates
        keypoints = torch.zeros(batch_size, num_keypoints*2).to(feature_maps.device)
        keypoints[:, 0::2] = x_mean 
        keypoints[:, 1::2] = y_mean 
        
        return keypoints, confidences