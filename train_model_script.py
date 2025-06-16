import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import yaml
import torch
import argparse
from src.data import create_data_loaders, DataAugmentation, calculate_class_weights, load_annotations
from src.models import create_model
from src.training import Trainer
from src.utils import DatabaseManager, Visualizer

def main():
    parser = argparse.ArgumentParser(description='Train X-ray object classification model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--experiment', type=str, default='resnet50_baseline', help='Experiment name')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    torch.manual_seed(config['seed'])
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(config['paths']['model_save'], exist_ok=True)
    os.makedirs(config['paths']['checkpoints'], exist_ok=True)
    
    annotations = load_annotations(config['data']['annotations_path'])
    class_weights = calculate_class_weights(annotations).to(device)
    
    augmentation = DataAugmentation(config)
    train_transform = augmentation.get_train_transforms()
    val_transform = augmentation.get_val_transforms()
    
    train_loader, val_loader, test_loader = create_data_loaders(
        config['data']['annotations_path'],
        config['data']['raw_path'],
        config,
        train_transform,
        val_transform
    )
    
    model = create_model(config)
    trainer = Trainer(model, config, device, class_weights)
    
    db = DatabaseManager()
    experiment_id = db.save_experiment(args.experiment, config)
    
    print(f"Starting training for experiment: {args.experiment}")
    history = trainer.train(train_loader, val_loader)
    
    for epoch, (train_loss, val_loss, train_acc, val_acc) in enumerate(
        zip(history['train_losses'], history['val_losses'], 
            history['train_accuracies'], history['val_accuracies'])
    ):
        db.save_training_result(experiment_id, epoch, train_loss, val_loss, train_acc, val_acc)
    
    visualizer = Visualizer(config['classes'])
    visualizer.plot_training_history(
        history, 
        save_path=f"{config['paths']['results']}/training_history_{args.experiment}.png"
    )
    
    print("Training completed!")
    print(f"Best model saved to: {config['paths']['model_save']}/best_model.pth")

if __name__ == "__main__":
    main()