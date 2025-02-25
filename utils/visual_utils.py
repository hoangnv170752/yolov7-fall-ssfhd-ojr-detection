import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time


def draw_keypoints(image, keypoints, occlusion_mask=None, skeleton=None, radius=4, thickness=2):
    """
    Draw keypoints on the image.
    
    Args:
        image: OpenCV image (BGR)
        keypoints: Keypoint coordinates (num_keypoints*2)
        occlusion_mask: Binary mask indicating occluded joints (1=occluded)
        skeleton: List of keypoint connections [(kpt_idx1, kpt_idx2), ...]
        radius: Radius of keypoint circles
        thickness: Thickness of skeleton lines
    """
    # Default COCO skeleton if none provided
    if skeleton is None:
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 6), (5, 11), (6, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
    
    # Extract x, y coordinates
    x_coords = keypoints[0::2]
    y_coords = keypoints[1::2]
    
    # Draw skeleton connections
    for connection in skeleton:
        i, j = connection
        if i < len(x_coords) and j < len(x_coords):
            if occlusion_mask is not None:
                # Skip if either keypoint is occluded
                if occlusion_mask[i*2] > 0 or occlusion_mask[j*2] > 0:
                    continue
            
            x1, y1 = int(x_coords[i]), int(y_coords[i])
            x2, y2 = int(x_coords[j]), int(y_coords[j])
            
            # Draw connection if both points are valid
            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), thickness)
    
    # Draw keypoints
    for i in range(len(x_coords)):
        x, y = int(x_coords[i]), int(y_coords[i])
        
        # Skip invalid keypoints
        if x <= 0 or y <= 0:
            continue
        
        # Different colors for visible and occluded keypoints
        if occlusion_mask is not None and occlusion_mask[i*2] > 0:
            # Occluded keypoint (red)
            cv2.circle(image, (x, y), radius, (0, 0, 255), -1)
        else:
            # Visible keypoint (green)
            cv2.circle(image, (x, y), radius, (0, 255, 0), -1)
    
    return image


def draw_fall_alert(image, image_size, font=cv2.FONT_HERSHEY_SIMPLEX, scale=1.5, thickness=2):
    """
    Draw fall alert on the image.
    
    Args:
        image: OpenCV image (BGR)
        image_size: Tuple of (width, height)
        font: OpenCV font
        scale: Font scale
        thickness: Text thickness
    """
    width, height = image_size
    text = "FALLLLLL!"
    
    # Calculate text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    
    # Calculate position (centered at the top)
    x = (width - text_width) // 2
    y = text_height + 20  # 20 pixels from the top
    
    # Draw red background box
    cv2.rectangle(
        image,
        (x - 10, y - text_height - 10),
        (x + text_width + 10, y + 10),
        (0, 0, 255),
        -1
    )
    
    # Draw text
    cv2.putText(
        image,
        text,
        (x, y),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )
    
    # Get current timestamp
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Draw timestamp
    cv2.putText(
        image,
        timestamp,
        (x, y + 30),
        font,
        0.7,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )
    
    return image


def create_heatmap(keypoints, image_shape, sigma=3):
    """
    Create heatmap visualization of keypoints.
    
    Args:
        keypoints: Keypoint coordinates (num_keypoints*2)
        image_shape: Shape of the target image (height, width)
        sigma: Gaussian sigma for keypoint spread
        
    Returns:
        heatmap: Heatmap visualization
    """
    height, width = image_shape[:2]
    num_keypoints = len(keypoints) // 2
    
    # Create empty heatmap
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    # Extract x, y coordinates
    x_coords = keypoints[0::2]
    y_coords = keypoints[1::2]
    
    # Generate meshgrid for faster computation
    y, x = np.mgrid[0:height, 0:width]
    
    # Add each keypoint to the heatmap
    for i in range(num_keypoints):
        kpt_x, kpt_y = int(x_coords[i]), int(y_coords[i])
        
        # Skip invalid keypoints
        if kpt_x <= 0 or kpt_y <= 0 or kpt_x >= width or kpt_y >= height:
            continue
        
        # Compute Gaussian
        gaussian = np.exp(-((x - kpt_x) ** 2 + (y - kpt_y) ** 2) / (2 * sigma ** 2))
        
        # Add to heatmap
        heatmap = np.maximum(heatmap, gaussian)
    
    # Normalize to 0-1
    heatmap = heatmap / np.max(heatmap) if np.max(heatmap) > 0 else heatmap
    
    # Convert to color map
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    return heatmap_color


def visualize_fall_detection(frames, keypoints, fall_probs, occlusion_masks=None, output_path=None):
    """
    Visualize fall detection results.
    
    Args:
        frames: List of frames (numpy arrays)
        keypoints: List of keypoint arrays
        fall_probs: List of fall probabilities
        occlusion_masks: List of occlusion masks
        output_path: Path to save visualization
    """
    num_frames = len(frames)
    
    # Create figure
    fig, axes = plt.subplots(num_frames, 2, figsize=(12, 3*num_frames))
    
    # If single frame, convert axes to 2D array
    if num_frames == 1:
        axes = axes.reshape(1, -1)
    
    # Process each frame
    for i in range(num_frames):
        frame = frames[i]
        kpts = keypoints[i]
        prob = fall_probs[i]
        mask = occlusion_masks[i] if occlusion_masks is not None else None
        
        # Draw original frame with keypoints
        frame_with_kpts = frame.copy()
        draw_keypoints(frame_with_kpts, kpts, mask)
        
        # Show frame with keypoints
        axes[i, 0].imshow(cv2.cvtColor(frame_with_kpts, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"Frame {i+1}")
        axes[i, 0].axis('off')
        
        # Show heatmap visualization
        heatmap = create_heatmap(kpts, frame.shape[:2])
        axes[i, 1].imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title(f"Fall Probability: {prob:.2f}")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    
    # Save or show figure
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def create_comparison_video(original_video, detection_results, output_path):
    """
    Create a side-by-side comparison video.
    
    Args:
        original_video: Path to original video
        detection_results: Path to detection results video
        output_path: Path to save comparison video
    """
    # Open videos
    cap_original = cv2.VideoCapture(original_video)
    cap_detection = cv2.VideoCapture(detection_results)
    
    # Check if videos opened successfully
    if not cap_original.isOpened() or not cap_detection.isOpened():
        print("Error: Could not open videos")
        return
    
    # Get video properties
    width_orig = int(cap_original.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_orig = int(cap_original.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width_det = int(cap_detection.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_det = int(cap_detection.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap_original.get(cv2.CAP_PROP_FPS))
    
    # Calculate combined width
    combined_width = width_orig + width_det
    combined_height = max(height_orig, height_det)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, combined_height))
    
    while True:
        # Read frames
        ret_orig, frame_orig = cap_original.read()
        ret_det, frame_det = cap_detection.read()
        
        # Break if either video ends
        if not ret_orig or not ret_det:
            break
        
        # Resize frames to same height if needed
        if height_orig != height_det:
            if height_orig > height_det:
                frame_det = cv2.resize(frame_det, (int(width_det * combined_height / height_det), combined_height))
            else:
                frame_orig = cv2.resize(frame_orig, (int(width_orig * combined_height / height_orig), combined_height))
        
        # Combine frames side by side
        combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        combined_frame[:, :width_orig] = frame_orig
        combined_frame[:, width_orig:] = frame_det
        
        # Add labels
        cv2.putText(combined_frame, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined_frame, "Detection", (width_orig + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Write frame
        out.write(combined_frame)
    
    # Release resources
    cap_original.release()
    cap_detection.release()
    out.release()
    
    print(f"Comparison video saved to {output_path}")


def visualize_occlusion_recovery(before_frame, after_frame, before_keypoints, after_keypoints, 
                               occlusion_mask, output_path=None):
    """
    Visualize occlusion recovery results.
    
    Args:
        before_frame: Frame with occlusions
        after_frame: Same frame after recovery
        before_keypoints: Keypoints before recovery
        after_keypoints: Keypoints after recovery
        occlusion_mask: Mask indicating occluded joints
        output_path: Path to save visualization
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Draw before recovery
    before_img = before_frame.copy()
    draw_keypoints(before_img, before_keypoints, occlusion_mask)
    ax1.imshow(cv2.cvtColor(before_img, cv2.COLOR_BGR2RGB))
    ax1.set_title("Before Occlusion Recovery")
    ax1.axis('off')
    
    # Draw after recovery
    after_img = after_frame.copy()
    draw_keypoints(after_img, after_keypoints, None)
    ax2.imshow(cv2.cvtColor(after_img, cv2.COLOR_BGR2RGB))
    ax2.set_title("After Occlusion Recovery")
    ax2.axis('off')
    
    plt.tight_layout()
    
    # Save or show figure
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()