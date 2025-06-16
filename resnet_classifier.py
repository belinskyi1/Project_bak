import torch
import torch.nn as nn
from torchvision import models

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=5, pretrained=True, dropout=0.5):
        super(ResNetClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes)
        )
        
    def forward(self, x):
        return self.resnet(x)

def create_model(config):
    model = ResNetClassifier(
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        dropout=config['model']['dropout']
    )
    return model

def load_model(model_path, config, device):
    model = create_model(config)
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def save_model(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)