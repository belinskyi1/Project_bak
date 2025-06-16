import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

class SIXrayDataset(Dataset):
    def __init__(self, annotations, image_dir, transform=None, class_to_idx=None):
        self.annotations = annotations
        self.image_dir = image_dir
        self.transform = transform
        self.class_to_idx = class_to_idx or {
            'pliers': 0, 'gun': 1, 'wrench': 2, 'knife': 3, 'scissors': 4
        }
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        image_path = os.path.join(self.image_dir, row['image_name'])
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = self.class_to_idx[row['class']]
        return image, torch.tensor(label, dtype=torch.long)

def load_annotations(annotations_path):
    annotations = []
    for file in os.listdir(annotations_path):
        if file.endswith('.json'):
            with open(os.path.join(annotations_path, file), 'r') as f:
                data = json.load(f)
                for obj in data.get('objects', []):
                    annotations.append({
                        'image_name': data['image_name'],
                        'class': obj['class'],
                        'bbox': obj['bbox']
                    })
    return pd.DataFrame(annotations)

def create_data_loaders(annotations_path, image_dir, config, transform_train=None, transform_val=None):
    annotations = load_annotations(annotations_path)
    
    train_data, temp_data = train_test_split(
        annotations, test_size=(1 - config['data']['train_split']), 
        random_state=config['seed'], stratify=annotations['class']
    )
    
    val_size = config['data']['val_split'] / (1 - config['data']['train_split'])
    val_data, test_data = train_test_split(
        temp_data, test_size=(1 - val_size),
        random_state=config['seed'], stratify=temp_data['class']
    )
    
    train_dataset = SIXrayDataset(train_data, image_dir, transform_train)
    val_dataset = SIXrayDataset(val_data, image_dir, transform_val)
    test_dataset = SIXrayDataset(test_data, image_dir, transform_val)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config['training']['batch_size'],
        shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['training']['batch_size'],
        shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['training']['batch_size'],
        shuffle=False, num_workers=4
    )
    
    return train_loader, val_loader, test_loader