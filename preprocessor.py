import cv2
import numpy as np
from torchvision import transforms

class ImagePreprocessor:
    def __init__(self, config):
        self.image_size = tuple(config['preprocessing']['image_size'])
        self.normalize = config['preprocessing']['normalize']
        self.enhance_contrast = config['preprocessing']['enhance_contrast']
        self.gaussian_sigma = config['preprocessing']['gaussian_sigma']
        
    def resize_image(self, image):
        return cv2.resize(image, self.image_size)
    
    def enhance_contrast_clahe(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    def apply_gaussian_filter(self, image):
        kernel_size = int(6 * self.gaussian_sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), self.gaussian_sigma)
    
    def preprocess_image(self, image):
        if isinstance(image, np.ndarray):
            processed = image.copy()
        else:
            processed = np.array(image)
            
        processed = self.resize_image(processed)
        
        if self.enhance_contrast:
            processed = self.enhance_contrast_clahe(processed)
            
        if self.gaussian_sigma > 0:
            processed = self.apply_gaussian_filter(processed)
            
        if len(processed.shape) == 2:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            
        return processed

def get_transforms(config, is_train=False):
    image_size = tuple(config['preprocessing']['image_size'])
    
    transform_list = [
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ]
    
    if config['preprocessing']['normalize']:
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    
    return transforms.Compose(transform_list)