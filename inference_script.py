import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import yaml
import torch
import cv2
import argparse
import numpy as np
from PIL import Image
from src.models import load_model, ObjectDetector
from src.data import ImagePreprocessor

def main():
    parser = argparse.ArgumentParser(description='Run inference on X-ray image')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, help='Path to save result image')
    parser.add_argument('--threshold', type=float, default=0.7, help='Confidence threshold')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    model = load_model(args.model, config, device)
    detector = ObjectDetector(confidence_threshold=args.threshold)
    preprocessor = ImagePreprocessor(config)
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found - {args.image}")
        return
    
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image - {args.image}")
        return
    
    print(f"Processing image: {args.image}")
    
    processed_image = preprocessor.preprocess_image(image)
    detections = detector.detect_objects(processed_image, model, device, config['classes'])
    
    if detections:
        print(f"Found {len(detections)} objects:")
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            print(f"  {i+1}. {class_name} (confidence: {confidence:.3f}) at [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
            
        if args.output:
            result_image = draw_detections(image.copy(), detections)
            cv2.imwrite(args.output, result_image)
            print(f"Result saved to: {args.output}")
    else:
        print("No objects detected above threshold.")

def draw_detections(image, detections):
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    for i, detection in enumerate(detections):
        bbox = detection['bbox']
        class_name = detection['class']
        confidence = detection['confidence']
        color = colors[i % len(colors)]
        
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return image

if __name__ == "__main__":
    main()