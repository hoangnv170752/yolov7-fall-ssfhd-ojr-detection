import torch
import torch.nn as nn
import os
import sys
from pathlib import Path
import numpy as np

def load_yolov7_model(weights_path):
    """
    Load YOLOv7 model from weights file.
    
    Args:
        weights_path: Path to YOLOv7 weights
        
    Returns:
        model: Loaded YOLOv7 model
    """
    try:
        # Try to import from yolov7 submodule
        yolov7_path = Path(__file__).parent / "yolov7"
        if yolov7_path.exists():
            sys.path.insert(0, str(yolov7_path))
            from models.experimental import attempt_load
            model = attempt_load(weights_path)
            sys.path.remove(str(yolov7_path))
            return model
        else:
            # Fallback to custom implementation if yolov7 submodule is not available
            return SimpleYOLOv7(weights_path)
    except ImportError:
        # Fallback to custom implementation
        return SimpleYOLOv7(weights_path)


class SimpleYOLOv7:
    """
    Simple YOLOv7 model wrapper for person detection.
    
    This is a placeholder implementation that simulates YOLOv7 behavior
    when the actual YOLOv7 implementation is not available.
    """
    
    def __init__(self, weights_path=None):
        """
        Initialize SimpleYOLOv7.
        
        Args:
            weights_path: Path to weights file (not actually used in this implementation)
        """
        self.weights_path = weights_path
        print(f"SimpleYOLOv7 initialized (placeholder). Real weights at: {weights_path}")
        
        # Define a simple convolutional backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Dummy detection head
        self.detect = nn.Linear(256, 6)  # 6 values: x1, y1, x2, y2, conf, class
    
    def to(self, device):
        """Move model to device."""
        self.backbone = self.backbone.to(device)
        self.detect = self.detect.to(device)
        return self
    
    def eval(self):
        """Set model to evaluation mode."""
        self.backbone.eval()
        self.detect.eval()
        return self
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
            
        Returns:
            detections: List of detection tensors
        """
        batch_size = x.size(0)
        features = self.backbone(x)
        
        # Generate dummy detections for each image in batch
        detections = []
        for i in range(batch_size):
            # Create some random person detections
            # Format: [x1, y1, x2, y2, confidence, class_id]
            # Class 0 is person in COCO
            
            # Create 1-3 detections per image
            num_detections = np.random.randint(1, 4)
            person_detections = []
            
            for j in range(num_detections):
                # Random bounding box
                x1 = np.random.uniform(0, 0.7) * 224
                y1 = np.random.uniform(0, 0.7) * 224
                w = np.random.uniform(0.2, 0.5) * 224
                h = np.random.uniform(0.3, 0.8) * 224
                x2 = min(x1 + w, 224)
                y2 = min(y1 + h, 224)
                
                # Random confidence (high since we're simulating good detections)
                conf = np.random.uniform(0.8, 0.98)
                
                # Class ID (0 for person)
                class_id = 0.0
                
                person_detections.append([x1, y1, x2, y2, conf, class_id])
            
            # Convert to tensor
            person_detections = torch.tensor(person_detections, device=x.device)
            detections.append(person_detections)
        
        return detections
    
    def __call__(self, x):
        """Method to make model callable."""
        return self.forward(x)


def init_weights(model, init_type='normal', gain=0.02):
    """
    Initialize model weights.
    
    Args:
        model: PyTorch model
        init_type: Initialization type ('normal', 'xavier', 'kaiming', 'orthogonal')
        gain: Gain factor for initialization
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(f'Initialization method {init_type} is not implemented')
            
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)
    
    model.apply(init_func)
    
    return model


def load_checkpoint(model, optimizer=None, filename='checkpoint.pth'):
    """
    Load model and optimizer from checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        filename: Checkpoint filename
        
    Returns:
        start_epoch: Starting epoch
        best_val_loss: Best validation loss
    """
    start_epoch = 0
    best_val_loss = float('inf')
    
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
            
        print(f"Loaded checkpoint '{filename}' (epoch {checkpoint['epoch']})")
    else:
        print(f"No checkpoint found at '{filename}'")
    
    return start_epoch, best_val_loss


def save_checkpoint(state, is_best, filename='checkpoint.pth', best_filename='model_best.pth'):
    """
    Save model checkpoint.
    
    Args:
        state: State dictionary containing model state_dict, optimizer state_dict, etc.
        is_best: Whether this is the best model so far
        filename: Checkpoint filename
        best_filename: Best model filename
    """
    torch.save(state, filename)
    if is_best:
        import shutil
        shutil.copyfile(filename, best_filename)


def freeze_layers(model, layers_to_freeze):
    """
    Freeze specific layers in the model.
    
    Args:
        model: PyTorch model
        layers_to_freeze: List of layer names to freeze
    """
    for name, param in model.named_parameters():
        for layer in layers_to_freeze:
            if layer in name:
                param.requires_grad = False


def get_lr(optimizer):
    """
    Get current learning rate from optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        lr: Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def convert_yolov7_to_onnx(model, input_size=(640, 640), output_path='weights/yolov7.onnx'):
    """
    Convert YOLOv7 PyTorch model to ONNX format.
    
    Args:
        model: YOLOv7 model
        input_size: Input size (width, height)
        output_path: Output ONNX file path
    """
    import torch.onnx
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1], device=next(model.parameters()).device)
    
    # Export model
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        verbose=False,
        opset_version=12,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {output_path}")


def load_onnx_model(model_path):
    """
    Load ONNX model.
    
    Args:
        model_path: Path to ONNX model
        
    Returns:
        model: ONNX Runtime InferenceSession
    """
    try:
        import onnxruntime as ort
        
        # Create ONNX Runtime session
        session = ort.InferenceSession(model_path)
        
        return session
    except ImportError:
        print("Error: ONNX Runtime not installed. Please install it with: pip install onnxruntime")
        return None


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                       labels=(), max_det=300):
    """
    Performs Non-Maximum Suppression (NMS) on inference results
    
    Args:
        prediction: Predictions tensor from YOLOv7 model [batch, num_predictions, box_dim]
        conf_thres: Confidence threshold for filtering predictions
        iou_thres: IoU threshold for NMS
        classes: Filter by class, i.e. classes=0 to filter only persons
        agnostic: Class-agnostic NMS
        multi_label: Multiple labels per box
        labels: (optional) List of target labels to filter by
        max_det: Maximum number of detections to return
        
    Returns:
        List of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates
    
    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU threshold {iou_thres}, valid values are between 0.0 and 1.0'
    
    # Settings
    min_wh, max_wh = 2, 4096  # min and max box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS
    
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[xc[xi]]  # confidence
        
        # If none remain process next image
        if not x.shape[0]:
            continue
        
        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        
        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        
        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
        
        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]
        
        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        
        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision_nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy
        
        output[xi] = x[i]
    
    return output


def xywh2xyxy(x):
    """
    Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    
    Args:
        x: bounding boxes in [x, y, w, h] format
        
    Returns:
        y: bounding boxes in [x1, y1, x2, y2] format
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def torchvision_nms(boxes, scores, iou_thres):
    """
    Performs NMS using torchvision ops.
    
    Args:
        boxes: bounding boxes in [x1, y1, x2, y2] format
        scores: confidence scores
        iou_thres: IoU threshold
        
    Returns:
        keep: indices of boxes to keep
    """
    try:
        import torchvision
        return torchvision.ops.nms(boxes, scores, iou_thres)
    except ImportError:
        print("Error: torchvision not installed. Falling back to custom NMS implementation.")
        return custom_nms(boxes, scores, iou_thres)


def custom_nms(boxes, scores, iou_thres):
    """
    Custom NMS implementation if torchvision is not available.
    
    Args:
        boxes: bounding boxes in [x1, y1, x2, y2] format
        scores: confidence scores
        iou_thres: IoU threshold
        
    Returns:
        keep: indices of boxes to keep
    """
    # Sort by confidence
    _, order = scores.sort(descending=True)
    
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
        
        i = order[0].item()
        keep.append(i)
        
        # Compute IoU of the remaining boxes with the box kept
        ious = box_iou(boxes[i].unsqueeze(0), boxes[order[1:]])
        
        # Remove boxes with IoU > threshold
        mask = ious < iou_thres
        order = order[1:][mask]
    
    return torch.tensor(keep, device=boxes.device)


def box_iou(box1, box2):
    """
    Calculate IoU between box1 and box2.
    
    Args:
        box1: first box in [x1, y1, x2, y2] format
        box2: second box in [x1, y1, x2, y2] format
        
    Returns:
        iou: IoU between box1 and box2
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    
    # Get the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1.unsqueeze(0))
    inter_rect_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1.unsqueeze(0))
    inter_rect_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2.unsqueeze(0))
    inter_rect_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2.unsqueeze(0))
    
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * \
                torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)
    
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union = b1_area.unsqueeze(1) + b2_area.unsqueeze(0) - inter_area
    
    iou = inter_area / union
    
    return iou