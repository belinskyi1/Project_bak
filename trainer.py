import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, model, config, device, class_weights=None):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        self.best_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader):
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []
        
        for epoch in range(self.config['training']['epochs']):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            self.scheduler.step(val_loss)
            
            print(f'Epoch {epoch+1}/{self.config["training"]["epochs"]}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, train_loss, val_loss, is_best=True)
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.config['training']['early_stopping_patience']:
                print(f'Early stopping at epoch {epoch+1}')
                break
                
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies
        }
    
    def save_checkpoint(self, epoch, train_loss, val_loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_loss': self.best_loss
        }
        
        checkpoint_path = os.path.join(self.config['paths']['checkpoints'], f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.config['paths']['model_save'], 'best_model.pth')
            torch.save(checkpoint, best_path)