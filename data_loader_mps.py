import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
import random
from PIL import Image
import platform


class CutoutTransform:
    """Custom implementation of Cutout/CoarseDropout using torchvision"""
    def __init__(self, n_holes=1, length=16, fill_value=None):
        self.n_holes = n_holes
        self.length = length
        self.fill_value = fill_value
        
    def __call__(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to apply cutout
        Returns:
            PIL Image or Tensor: Image with cutout applied
        """
        if isinstance(img, Image.Image):
            img = np.array(img)
            was_pil = True
        else:
            was_pil = False
            
        h, w = img.shape[:2]
        
        # Use dataset mean if no fill value provided
        if self.fill_value is None:
            self.fill_value = [125, 122, 113]  # CIFAR-10 approximate means in 0-255 range
        
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)
            
            # Apply cutout
            img[y1:y2, x1:x2] = self.fill_value
        
        if was_pil:
            return Image.fromarray(img)
        return img


class ShiftScaleRotate:
    """Custom implementation of ShiftScaleRotate using torchvision"""
    def __init__(self, shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5):
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        self.p = p
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
            
        # Random parameters
        angle = random.uniform(-self.rotate_limit, self.rotate_limit)
        scale = random.uniform(1 - self.scale_limit, 1 + self.scale_limit)
        
        # Get image dimensions
        width, height = img.size if isinstance(img, Image.Image) else (img.shape[1], img.shape[0])
        
        # Calculate shift
        max_dx = self.shift_limit * width
        max_dy = self.shift_limit * height
        dx = random.uniform(-max_dx, max_dx)
        dy = random.uniform(-max_dy, max_dy)
        
        # Apply transformations using torchvision
        if isinstance(img, Image.Image):
            # Apply affine transformation
            img = transforms.functional.affine(
                img, 
                angle=angle,
                translate=(dx, dy),
                scale=scale,
                shear=0
            )
        
        return img


def get_transforms_mps(config):
    """Get training and validation transforms compatible with MPS"""
    
    # Calculate fill value for cutout (convert normalized mean back to 0-255 range)
    fill_value = tuple([int(m * 255) for m in config.MEAN])
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),
        CutoutTransform(
            n_holes=1,
            length=16,
            fill_value=fill_value
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_transform


def get_data_loaders_mps(config):
    """Create MPS-compatible data loaders"""
    
    # Detect if on Mac with MPS
    is_mac = platform.system() == 'Darwin'
    has_mps = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    
    # Set num_workers to 0 for MPS to avoid multiprocessing issues
    num_workers = 0 if (is_mac or has_mps) else 4
    
    # Disable pin_memory for MPS
    pin_memory = False if (is_mac or has_mps) else True
    
    if is_mac or has_mps:
        print(f"Detected Mac/MPS environment. Using num_workers={num_workers}, pin_memory={pin_memory}")
    
    train_transform, val_transform = get_transforms_mps(config)
    
    train_dataset = datasets.CIFAR10(
        root=config.DATA_PATH,
        train=True,
        transform=train_transform,
        download=True
    )
    
    val_dataset = datasets.CIFAR10(
        root=config.DATA_PATH,
        train=False,
        transform=val_transform,
        download=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader
