import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import os


class Trainer:
    """Trainer class for the CNN model"""
    
    def __init__(self, model, config):
        self.model = model.to(config.DEVICE)
        self.config = config
        self.device = config.DEVICE
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY
        )
        
        self.best_accuracy = 0
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss / len(pbar),
                'acc': 100. * correct / total
            })
        
        return running_loss / len(train_loader), 100. * correct / total
    
    def train_epoch_with_scheduler(self, train_loader):
        """Train for one epoch with scheduler step"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            # Step scheduler after each batch
            if hasattr(self, 'scheduler'):
                self.scheduler.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss / len(pbar),
                'acc': 100. * correct / total
            })
        
        return running_loss / len(train_loader), 100. * correct / total
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': running_loss / len(pbar),
                    'acc': 100. * correct / total
                })
        
        accuracy = 100. * correct / total
        return running_loss / len(val_loader), accuracy
    
    def train(self, train_loader, val_loader, num_epochs):
        """Full training loop"""
        
        # Setup learning rate scheduler
        total_steps = num_epochs * len(train_loader)
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=0.1,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            
            # Train
            train_loss, train_acc = self.train_epoch_with_scheduler(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self.save_checkpoint(epoch, val_acc)
                print(f'Best model saved! Accuracy: {val_acc:.2f}%')
    
    def save_checkpoint(self, epoch, accuracy):
        """Save model checkpoint"""
        os.makedirs(self.config.CHECKPOINT_PATH, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
        }
        
        path = os.path.join(self.config.CHECKPOINT_PATH, f'best_model.pth')
        torch.save(checkpoint, path)
