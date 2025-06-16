from torchvision import transforms
import torch

class DataAugmentation:
    def __init__(self, config):
        self.image_size = tuple(config['preprocessing']['image_size'])
        
    def get_train_transforms(self):
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def get_val_transforms(self):
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def calculate_class_weights(annotations, num_classes=5):
    class_counts = annotations['class'].value_counts()
    total_samples = len(annotations)
    weights = []
    
    class_names = ['pliers', 'gun', 'wrench', 'knife', 'scissors']
    for class_name in class_names:
        count = class_counts.get(class_name, 1)
        weight = total_samples / (num_classes * count)
        weights.append(weight)
    
    return torch.FloatTensor(weights)