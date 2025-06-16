import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Visualizer:
    def __init__(self, class_names):
        self.class_names = class_names
        plt.style.use('default')
        
    def plot_training_history(self, history, save_path=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = range(1, len(history['train_losses']) + 1)
        
        ax1.plot(epochs, history['train_losses'], 'b-', label='Training Loss')
        ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(epochs, history['train_accuracies'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, history['val_accuracies'], 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_confusion_matrix(self, cm, normalize=False, save_path=None):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
            
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt=fmt, cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_class_distribution(self, annotations, save_path=None):
        class_counts = annotations['class'].value_counts()
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(class_counts.index, class_counts.values)
        plt.title('Class Distribution in Dataset')
        plt.xlabel('Object Classes')
        plt.ylabel('Number of Annotations')
        plt.xticks(rotation=45)
        
        for bar, count in zip(bars, class_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_metrics_comparison(self, metrics, save_path=None):
        classes = self.class_names
        precision = metrics['precision_per_class']
        recall = metrics['recall_per_class']
        f1 = metrics['f1_per_class']
        
        x = np.arange(len(classes))
        width = 0.25
        
        plt.figure(figsize=(12, 6))
        plt.bar(x - width, precision, width, label='Precision', alpha=0.8)
        plt.bar(x, recall, width, label='Recall', alpha=0.8)
        plt.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        plt.xlabel('Classes')
        plt.ylabel('Score')
        plt.title('Performance Metrics by Class')
        plt.xticks(x, classes, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_learning_curve(self, train_sizes, train_scores, val_scores, save_path=None):
        plt.figure(figsize=(8, 6))
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.plot(train_sizes, train_mean, 'o-', label='Training Accuracy')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        
        plt.plot(train_sizes, val_mean, 'o-', label='Validation Accuracy')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
        
        plt.xlabel('Training Set Size (%)')
        plt.ylabel('Accuracy')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()