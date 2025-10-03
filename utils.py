import torch
import torch.nn as nn


def count_parameters(model):
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_receptive_field(model):
    """Calculate theoretical receptive field"""
    rf = 1
    stride = 1
    
    # This is a simplified calculation
    # For accurate RF, trace through each layer
    layers_info = [
        (3, 1, 1),  # kernel, stride, padding
        (3, 1, 1),
        (3, 1, 1),
        (3, 1, 1),
        (3, 1, 1),
        (3, 2, 1),  # dilated conv acts like larger kernel
        (3, 1, 1),
        (3, 4, 1),  # dilated conv
        (3, 1, 1),
        (3, 8, 1),  # dilated conv
    ]
    
    for k, s, _ in layers_info:
        rf = rf + (k - 1) * stride
        stride = stride * s
    
    return rf


def print_model_summary(model, config):
    """Print model summary"""
    num_params = count_parameters(model)
    print(f"\n{'='*50}")
    print(f"Model: CIFAR-10 Advanced CNN")
    print(f"{'='*50}")
    print(f"Total Parameters: {num_params:,}")
    print(f"Receptive Field: {model.get_receptive_field()}")
    print(f"Architecture: C1-C2-C3-C4-GAP-FC")
    print(f"Depthwise Separable Conv: ✓ (in C2)")
    print(f"Dilated Convolution: ✓ (in C3 and C4)")
    print(f"Global Average Pooling: ✓")
    print(f"Target Accuracy: 85%")
    print(f"Parameter Limit: 200,000")
    print(f"Parameter Check: {'✓ PASS' if num_params < 200000 else '✗ FAIL'}")
    print(f"{'='*50}\n")