import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix

class MetricsCalculator:
    def __init__(self, class_names):
        self.class_names = class_names
        self.num_classes = len(class_names)
        
    def calculate_accuracy(self, predictions, targets):
        return np.mean(predictions == targets)
    
    def calculate_precision_recall_f1(self, predictions, targets):
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )
        
        avg_precision = np.mean(precision)
        avg_recall = np.mean(recall)
        avg_f1 = np.mean(f1)
        
        return {
            'precision_per_class': precision,
            'recall_per_class': recall,
            'f1_per_class': f1,
            'support_per_class': support,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1': avg_f1
        }
    
    def calculate_auc_roc(self, probabilities, targets):
        auc_scores = []
        for i in range(self.num_classes):
            binary_targets = (targets == i).astype(int)
            class_probs = probabilities[:, i]
            try:
                auc = roc_auc_score(binary_targets, class_probs)
                auc_scores.append(auc)
            except ValueError:
                auc_scores.append(0.0)
        
        return auc_scores
    
    def generate_confusion_matrix(self, predictions, targets):
        cm = confusion_matrix(targets, predictions)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        return {
            'confusion_matrix': cm,
            'normalized_confusion_matrix': cm_normalized
        }
    
    def calculate_all_metrics(self, predictions, targets, probabilities=None):
        accuracy = self.calculate_accuracy(predictions, targets)
        prf_metrics = self.calculate_precision_recall_f1(predictions, targets)
        cm_data = self.generate_confusion_matrix(predictions, targets)
        
        results = {
            'accuracy': accuracy,
            **prf_metrics,
            **cm_data
        }
        
        if probabilities is not None:
            auc_scores = self.calculate_auc_roc(probabilities, targets)
            results['auc_per_class'] = auc_scores
            results['avg_auc'] = np.mean(auc_scores)
        
        return results
    
    def format_results(self, metrics):
        formatted = f"Overall Accuracy: {metrics['accuracy']:.3f}\n\n"
        
        formatted += "Per-class metrics:\n"
        for i, class_name in enumerate(self.class_names):
            precision = metrics['precision_per_class'][i]
            recall = metrics['recall_per_class'][i]
            f1 = metrics['f1_per_class'][i]
            
            formatted += f"{class_name:>12}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}\n"
        
        formatted += f"\nAverage: P={metrics['avg_precision']:.3f}, "
        formatted += f"R={metrics['avg_recall']:.3f}, F1={metrics['avg_f1']:.3f}"
        
        if 'avg_auc' in metrics:
            formatted += f", AUC={metrics['avg_auc']:.3f}"
        
        return formatted