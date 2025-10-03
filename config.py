import torch

class Config:
    # Dataset
    DATASET = 'CIFAR-10'
    NUM_CLASSES = 10
    IMAGE_SIZE = 32
    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.2470, 0.2435, 0.2616)
    
    # Model Architecture
    INITIAL_CHANNELS = 3
    
    # Training
    BATCH_SIZE = 128
    LEARNING_RATE = 0.01
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 50
    
    # Augmentation
    USE_AUGMENTATION = True
    
    # Device
    if torch.backends.mps.is_available():
        DEVICE = 'mps'
    elif torch.cuda.is_available():
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'
    
    # Paths
    DATA_PATH = './data'
    CHECKPOINT_PATH = './checkpoints'
