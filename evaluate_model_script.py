import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import yaml
import torch
import argparse
from src.data import create_data_loaders, DataAugmentation, load_annotations
from src.models import load_model
from src.training import Validator
from src.utils import MetricsCalculator, Visualizer, DatabaseManager

def main():
    parser = argparse.ArgumentParser(description='Evaluate X-ray object classification model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--experiment_id', type=int, help='Experiment ID for database storage')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    annotations = load_annotations(config['data']['annotations_path'])
    
    augmentation = DataAugmentation(config)
    val_transform = augmentation.get_val_transforms()
    
    _, _, test_loader = create_data_loaders(
        config['data']['annotations_path'],
        config['data']['raw_path'],
        config,
        val_transform,
        val_transform
    )
    
    model = load_model(args.model, config, device)
    
    validator = Validator(model, device, config['classes'])
    predictions, targets, probabilities = validator.evaluate(test_loader)
    
    metrics_calc = MetricsCalculator(config['classes'])
    metrics = metrics_calc.calculate_all_metrics(predictions, targets, probabilities)
    
    print("Evaluation Results:")
    print("=" * 50)
    print(metrics_calc.format_results(metrics))
    
    visualizer = Visualizer(config['classes'])
    
    os.makedirs(config['paths']['results'], exist_ok=True)
    
    visualizer.plot_confusion_matrix(
        metrics['confusion_matrix'],
        save_path=f"{config['paths']['results']}/confusion_matrix.png"
    )
    
    visualizer.plot_confusion_matrix(
        metrics['confusion_matrix'], 
        normalize=True,
        save_path=f"{config['paths']['results']}/confusion_matrix_normalized.png"
    )
    
    visualizer.plot_metrics_comparison(
        metrics,
        save_path=f"{config['paths']['results']}/metrics_comparison.png"
    )
    
    object_count_results = validator.evaluate_by_object_count(test_loader, annotations)
    print("\nAccuracy by number of objects:")
    for count, accuracy in object_count_results.items():
        print(f"{count}: {accuracy:.3f}")
    
    if args.experiment_id:
        db = DatabaseManager()
        db.save_evaluation(args.experiment_id, metrics)
        print(f"\nResults saved to database for experiment {args.experiment_id}")

if __name__ == "__main__":
    main()