import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import yaml
import argparse
from tqdm import tqdm
import cv2
from src.data import ImagePreprocessor, load_annotations
from src.utils import Visualizer

def main():
    parser = argparse.ArgumentParser(description='Preprocess X-ray dataset')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory with raw images')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for processed images')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    preprocessor = ImagePreprocessor(config)
    
    annotations = load_annotations(config['data']['annotations_path'])
    unique_images = annotations['image_name'].unique()
    
    print(f"Processing {len(unique_images)} images...")
    
    processed_count = 0
    failed_count = 0
    
    for image_name in tqdm(unique_images, desc="Processing images"):
        input_path = os.path.join(args.input_dir, image_name)
        output_path = os.path.join(args.output_dir, image_name)
        
        if not os.path.exists(input_path):
            print(f"Warning: Image not found - {input_path}")
            failed_count += 1
            continue
            
        try:
            image = cv2.imread(input_path)
            if image is None:
                print(f"Warning: Could not load image - {input_path}")
                failed_count += 1
                continue
                
            processed_image = preprocessor.preprocess_image(image)
            cv2.imwrite(output_path, processed_image)
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {input_path}: {str(e)}")
            failed_count += 1
    
    print(f"\nPreprocessing completed!")
    print(f"Successfully processed: {processed_count} images")
    print(f"Failed: {failed_count} images")
    
    if processed_count > 0:
        visualizer = Visualizer(config['classes'])
        visualizer.plot_class_distribution(
            annotations,
            save_path=f"{config['paths']['results']}/class_distribution.png"
        )
        print("Class distribution plot saved.")

if __name__ == "__main__":
    main()