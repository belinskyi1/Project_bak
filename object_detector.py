import numpy as np
import cv2
from typing import List, Tuple

class ObjectDetector:
    def __init__(self, window_size=(224, 224), stride=50, confidence_threshold=0.7):
        self.window_size = window_size
        self.stride = stride
        self.confidence_threshold = confidence_threshold
        
    def sliding_window(self, image):
        h, w = image.shape[:2]
        windows = []
        positions = []
        
        for y in range(0, h - self.window_size[1] + 1, self.stride):
            for x in range(0, w - self.window_size[0] + 1, self.stride):
                window = image[y:y + self.window_size[1], x:x + self.window_size[0]]
                if window.shape[:2] == self.window_size:
                    windows.append(window)
                    positions.append((x, y))
                    
        return windows, positions
    
    def non_max_suppression(self, boxes, scores, iou_threshold=0.5):
        if len(boxes) == 0:
            return []
            
        boxes = np.array(boxes)
        scores = np.array(scores)
        
        indices = np.argsort(scores)[::-1]
        keep = []
        
        while len(indices) > 0:
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
                
            current_box = boxes[current]
            remaining_boxes = boxes[indices[1:]]
            
            ious = self.calculate_iou(current_box, remaining_boxes)
            indices = indices[1:][ious < iou_threshold]
            
        return keep
    
    def calculate_iou(self, box1, boxes):
        x1, y1, x2, y2 = box1
        xx1 = np.maximum(x1, boxes[:, 0])
        yy1 = np.maximum(y1, boxes[:, 1])
        xx2 = np.minimum(x2, boxes[:, 2])
        yy2 = np.minimum(y2, boxes[:, 3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        intersection = w * h
        
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area1 + area2 - intersection
        
        return intersection / union
    
    def detect_objects(self, image, model, device, class_names):
        windows, positions = self.sliding_window(image)
        if not windows:
            return []
            
        detections = []
        for window, (x, y) in zip(windows, positions):
            window_rgb = cv2.cvtColor(window, cv2.COLOR_BGR2RGB)
            window_tensor = self.preprocess_window(window_rgb, device)
            
            with torch.no_grad():
                outputs = model(window_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                if confidence.item() > self.confidence_threshold:
                    x2 = x + self.window_size[0]
                    y2 = y + self.window_size[1]
                    
                    detections.append({
                        'bbox': [x, y, x2, y2],
                        'class': class_names[predicted.item()],
                        'confidence': confidence.item()
                    })
        
        if detections:
            boxes = [d['bbox'] for d in detections]
            scores = [d['confidence'] for d in detections]
            keep_indices = self.non_max_suppression(boxes, scores)
            detections = [detections[i] for i in keep_indices]
            
        return detections
    
    def preprocess_window(self, window, device):
        import torch
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transform(window).unsqueeze(0).to(device)