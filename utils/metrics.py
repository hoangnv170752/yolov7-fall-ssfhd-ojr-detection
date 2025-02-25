import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix


def compute_accuracy(predictions, targets):
    """
    Compute accuracy from predictions and targets.
    
    Args:
        predictions: Predicted logits
        targets: Ground truth labels
        
    Returns:
        accuracy: Classification accuracy
    """
    if isinstance(predictions, torch.Tensor):
        # Get predicted class
        pred_class = torch.argmax(predictions, dim=1)
        correct = (pred_class == targets).float().sum()
        accuracy = correct / targets.size(0)
        return accuracy.item()
    else:
        # Numpy implementation
        pred_class = np.argmax(predictions, axis=1)
        correct = (pred_class == targets).sum()
        accuracy = correct / len(targets)
        return accuracy


def compute_precision_recall_f1(predictions, targets, threshold=0.5):
    """
    Compute precision, recall, and F1 score.
    
    Args:
        predictions: Predicted probabilities for positive class
        targets: Ground truth labels (binary)
        threshold: Classification threshold
        
    Returns:
        precision, recall, f1: Metric values
    """
    # Convert to numpy if tensors
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # Binary predictions
    binary_preds = (np.array(predictions) >= threshold).astype(int)
    targets = np.array(targets)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, binary_preds, average='binary', zero_division=0
    )
    
    return precision, recall, f1


def compute_auc(predictions, targets):
    """
    Compute Area Under the ROC Curve (AUC).
    
    Args:
        predictions: Predicted probabilities for positive class
        targets: Ground truth labels (binary)
        
    Returns:
        auc: AUC score
    """
    # Convert to numpy if tensors
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # Calculate AUC
    try:
        auc = roc_auc_score(targets, predictions)
    except ValueError:
        # Handle case where there's only one class in the targets
        auc = 0.5  # Default to random classifier
    
    return auc


def compute_confusion_matrix(predictions, targets, threshold=0.5):
    """
    Compute confusion matrix.
    
    Args:
        predictions: Predicted probabilities for positive class
        targets: Ground truth labels (binary)
        threshold: Classification threshold
        
    Returns:
        cm: Confusion matrix (TN, FP, FN, TP)
    """
    # Convert to numpy if tensors
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # Binary predictions
    binary_preds = (np.array(predictions) >= threshold).astype(int)
    targets = np.array(targets)
    
    # Calculate confusion matrix
    cm = confusion_matrix(targets, binary_preds)
    
    return cm


def compute_keypoint_metrics(pred_keypoints, gt_keypoints, occlusion_mask=None):
    """
    Compute keypoint estimation metrics.
    
    Args:
        pred_keypoints: Predicted keypoint coordinates (batch_size, num_keypoints*2)
        gt_keypoints: Ground truth keypoint coordinates (batch_size, num_keypoints*2)
        occlusion_mask: Binary mask of occluded keypoints (batch_size, num_keypoints*2)
        
    Returns:
        metrics: Dictionary of keypoint metrics (MPJPE, PCK, etc.)
    """
    # Convert to numpy if tensors
    if isinstance(pred_keypoints, torch.Tensor):
        pred_keypoints = pred_keypoints.cpu().numpy()
    if isinstance(gt_keypoints, torch.Tensor):
        gt_keypoints = gt_keypoints.cpu().numpy()
    if occlusion_mask is not None and isinstance(occlusion_mask, torch.Tensor):
        occlusion_mask = occlusion_mask.cpu().numpy()
    
    # Reshape to (batch_size, num_keypoints, 2)
    batch_size = pred_keypoints.shape[0]
    num_keypoints = pred_keypoints.shape[1] // 2
    
    pred_kpts = pred_keypoints.reshape(batch_size, num_keypoints, 2)
    gt_kpts = gt_keypoints.reshape(batch_size, num_keypoints, 2)
    
    # Calculate Euclidean distance for each keypoint
    distances = np.sqrt(np.sum((pred_kpts - gt_kpts) ** 2, axis=2))
    
    # Mean Per Joint Position Error (MPJPE)
    if occlusion_mask is not None:
        # Reshape occlusion mask
        mask = occlusion_mask.reshape(batch_size, num_keypoints, 2)
        mask = mask[:, :, 0]  # Use only x mask (same as y mask)
        
        # Only evaluate visible keypoints
        visible = (mask == 0)
        if visible.sum() > 0:
            mpjpe = np.mean(distances[visible])
        else:
            mpjpe = np.nan
    else:
        mpjpe = np.mean(distances)
    
    # Percentage of Correct Keypoints (PCK)
    # A keypoint is correct if its distance is less than a threshold
    # Threshold is typically a percentage of the person's bounding box size
    # Here we use a fixed threshold of 0.1 (normalized coordinates)
    threshold = 0.1
    
    if occlusion_mask is not None:
        if visible.sum() > 0:
            pck = np.mean((distances[visible] < threshold).astype(float))
        else:
            pck = np.nan
    else:
        pck = np.mean((distances < threshold).astype(float))
    
    return {
        'mpjpe': mpjpe,
        'pck': pck
    }


def compute_ojr_metrics(pred_keypoints, gt_keypoints, occlusion_mask):
    """
    Compute OJR performance metrics.
    
    Args:
        pred_keypoints: Predicted keypoint coordinates after recovery
        gt_keypoints: Ground truth keypoint coordinates
        occlusion_mask: Binary mask of occluded keypoints
        
    Returns:
        metrics: Dictionary of OJR metrics
    """
    # Convert to numpy if tensors
    if isinstance(pred_keypoints, torch.Tensor):
        pred_keypoints = pred_keypoints.cpu().numpy()
    if isinstance(gt_keypoints, torch.Tensor):
        gt_keypoints = gt_keypoints.cpu().numpy()
    if isinstance(occlusion_mask, torch.Tensor):
        occlusion_mask = occlusion_mask.cpu().numpy()
    
    # Ensure mask is binary
    mask = (occlusion_mask > 0).astype(bool)
    
    # Only evaluate occluded keypoints
    if mask.sum() == 0:
        return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'recovery_pck': np.nan}
    
    # Calculate metrics only for occluded keypoints
    mse = np.mean((pred_keypoints[mask] - gt_keypoints[mask]) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred_keypoints[mask] - gt_keypoints[mask]))
    
    # Calculate PCK for recovery
    # Reshape to (batch_size, num_keypoints, 2)
    batch_size = pred_keypoints.shape[0]
    num_keypoints = pred_keypoints.shape[1] // 2
    
    pred_kpts = pred_keypoints.reshape(batch_size, num_keypoints, 2)
    gt_kpts = gt_keypoints.reshape(batch_size, num_keypoints, 2)
    mask_reshaped = occlusion_mask.reshape(batch_size, num_keypoints, 2)
    mask_reshaped = mask_reshaped[:, :, 0]  # Use only x mask
    
    # Calculate distances for occluded keypoints
    distances = np.sqrt(np.sum((pred_kpts - gt_kpts) ** 2, axis=2))
    occluded = (mask_reshaped > 0)
    
    if occluded.sum() > 0:
        # PCK with threshold 0.1
        threshold = 0.1
        recovery_pck = np.mean((distances[occluded] < threshold).astype(float))
    else:
        recovery_pck = np.nan
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'recovery_pck': recovery_pck
    }


def evaluate_fall_detection(model, dataloader, device, threshold=0.5):
    """
    Evaluate fall detection model on a dataset.
    
    Args:
        model: SSHFD model
        dataloader: Test dataloader
        device: PyTorch device
        threshold: Classification threshold
        
    Returns:
        metrics: Dictionary of metrics
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for frames, labels in dataloader:
            # Move to device
            frames = frames.to(device)
            labels = labels.to(device)
            
            # Forward pass
            _, _, fall_preds = model(frames)
            
            # Get probabilities
            probs = torch.softmax(fall_preds, dim=1)[:, 1]  # Probability of fall
            
            # Store predictions and targets
            all_predictions.extend(probs.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    # Calculate metrics
    acc = compute_accuracy(np.stack([1-np.array(all_predictions), np.array(all_predictions)], axis=1), all_targets)
    precision, recall, f1 = compute_precision_recall_f1(all_predictions, all_targets, threshold)
    auc = compute_auc(all_predictions, all_targets)
    cm = compute_confusion_matrix(all_predictions, all_targets, threshold)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm
    }


def evaluate_ojr(ojr_model, sshfd_model, dataloader, device, occlusion_threshold=0.3):
    """
    Evaluate OJR model on a dataset.
    
    Args:
        ojr_model: OJR model
        sshfd_model: SSHFD model
        dataloader: Test dataloader
        device: PyTorch device
        occlusion_threshold: Threshold for synthetic occlusions
        
    Returns:
        metrics: Dictionary of metrics
    """
    ojr_model.eval()
    sshfd_model.eval()
    
    mse_values = []
    rmse_values = []
    mae_values = []
    pck_values = []
    
    with torch.no_grad():
        for frames, _ in dataloader:
            # Move to device
            frames = frames.to(device)
            
            # Get keypoints from SSHFD
            keypoints_seq, confidences_seq, _ = sshfd_model(frames)
            
            # Create synthetic occlusions
            batch_size = keypoints_seq.size(0)
            last_frame_keypoints = keypoints_seq[:, -1, :]
            
            # Random occlusion
            random_occlusion = torch.rand(batch_size, sshfd_model.num_keypoints).to(device)
            occlusion_mask = (random_occlusion < occlusion_threshold).float()
            
            # Expand mask
            expanded_mask = torch.zeros_like(last_frame_keypoints)
            expanded_mask[:, 0::2] = occlusion_mask
            expanded_mask[:, 1::2] = occlusion_mask
            
            # Apply occlusions
            occluded_keypoints = last_frame_keypoints.clone()
            occluded_keypoints[expanded_mask > 0] = 0
            
            # Create occluded sequence
            occluded_seq = keypoints_seq.clone()
            occluded_seq[:, -1, :] = occluded_keypoints
            
            # Recover keypoints
            recovered_keypoints = ojr_model(occluded_seq, expanded_mask)
            
            # Calculate metrics
            metrics = compute_ojr_metrics(
                recovered_keypoints.cpu().numpy(),
                last_frame_keypoints.cpu().numpy(),
                expanded_mask.cpu().numpy()
            )
            
            mse_values.append(metrics['mse'])
            rmse_values.append(metrics['rmse'])
            mae_values.append(metrics['mae'])
            pck_values.append(metrics['recovery_pck'])
    
    # Calculate average metrics
    mse_avg = np.nanmean(mse_values)
    rmse_avg = np.nanmean(rmse_values)
    mae_avg = np.nanmean(mae_values)
    pck_avg = np.nanmean(pck_values)
    
    return {
        'mse': mse_avg,
        'rmse': rmse_avg,
        'mae': mae_avg,
        'recovery_pck': pck_avg
    }