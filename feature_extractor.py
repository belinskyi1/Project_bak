import cv2
import numpy as np
from skimage.feature import hog

class FeatureExtractor:
    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        
    def compute_hog_features(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
        features = hog(
            image,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            visualize=False,
            feature_vector=True
        )
        return features
    
    def compute_gradient_map(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        return magnitude, direction
    
    def extract_texture_features(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
        glcm_features = self.compute_glcm_features(image)
        lbp_features = self.compute_lbp_features(image)
        
        return np.concatenate([glcm_features, lbp_features])
    
    def compute_glcm_features(self, image):
        from skimage.feature import greycomatrix, greycoprops
        
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        glcm = greycomatrix(
            image, distances=distances, angles=angles,
            levels=256, symmetric=True, normed=True
        )
        
        contrast = greycoprops(glcm, 'contrast').mean()
        dissimilarity = greycoprops(glcm, 'dissimilarity').mean()
        homogeneity = greycoprops(glcm, 'homogeneity').mean()
        energy = greycoprops(glcm, 'energy').mean()
        
        return np.array([contrast, dissimilarity, homogeneity, energy])
    
    def compute_lbp_features(self, image):
        from skimage.feature import local_binary_pattern
        
        radius = 3
        n_points = 8 * radius
        
        lbp = local_binary_pattern(image, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)
        
        return hist
    
    def extract_all_features(self, image):
        hog_features = self.compute_hog_features(image)
        texture_features = self.extract_texture_features(image)
        
        return np.concatenate([hog_features, texture_features])