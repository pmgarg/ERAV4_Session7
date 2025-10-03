import torch
import argparse
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from model import CIFAR10_CNN
from model_blocks import *
from data_loader_mps import get_data_loaders_mps
from trainer import Trainer
from utils import print_model_summary


def main(args):
    # Configuration
    config = Config()
    if args.epochs:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    
    # Detect and set device
    if torch.backends.mps.is_available():
        config.DEVICE = 'mps'
        print("Using MPS (Metal Performance Shaders) device")
    elif torch.cuda.is_available():
        config.DEVICE = 'cuda'
        print("Using CUDA device")
    else:
        config.DEVICE = 'cpu'
        print("Using CPU device")
    
    # Model
    model = CIFAR10_CNN(num_classes=config.NUM_CLASSES)
    print_model_summary(model, config)
    
    # Data loaders (MPS-compatible)
    train_loader, val_loader = get_data_loaders_mps(config)
    
    # Trainer
    trainer = Trainer(model, config)
    
    # Train
    trainer.train(train_loader, val_loader, config.NUM_EPOCHS)
    
    print(f"\n{'='*50}")
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {trainer.best_accuracy:.2f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CIFAR-10 CNN')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    
    args = parser.parse_args()
    main(args)
