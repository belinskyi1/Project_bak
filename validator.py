import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

class Validator:
    def __init__(self, model, device, class_names):
        self.model = model
        self.device = device
        self.class_names = class_names
        
    def evaluate(self, test_loader):
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Evaluating"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                probabilities = torch.softmax(output, dim=1)
                _, predicted = torch.max(output, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_targets), np.array(all_probabilities)
    
    def compute_metrics(self, predictions, targets):
        accuracy = np.mean(predictions == targets)
        
        cm = confusion_matrix(targets, predictions)
        report = classification_report(
            targets, predictions, 
            target_names=self.class_names, 
            output_dict=True
        )
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report
        }
    
    def evaluate_by_object_count(self, test_loader, annotations):
        predictions, targets, _ = self.evaluate(test_loader)
        
        image_counts = annotations.groupby('image_name').size()
        results_by_count = {}
        
        for count in range(1, 6):
            if count == 5:
                mask_images = image_counts >= count
            else:
                mask_images = image_counts == count
                
            if mask_images.sum() > 0:
                mask_indices = annotations[annotations['image_name'].isin(
                    image_counts[mask_images].index
                )].index
                
                if len(mask_indices) > 0:
                    count_predictions = predictions[mask_indices]
                    count_targets = targets[mask_indices]
                    accuracy = np.mean(count_predictions == count_targets)
                    results_by_count[f'{count}+ objects' if count == 5 else f'{count} object(s)'] = accuracy
        
        return results_by_count