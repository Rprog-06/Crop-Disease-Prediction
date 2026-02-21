# feature_extractor.py

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray

def extract_features(image_path):
    try:
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"[WARN] Could not read image: {image_path}")
            return None
        
        # Resize to a fixed size for consistency
        img = cv2.resize(img, (128, 128))
        
        # Convert to grayscale for texture analysis
        gray = rgb2gray(img)
        gray_scaled = (gray * 255).astype(np.uint8)

        # Texture features using Gray Level Co-occurrence Matrix (GLCM)
        glcm = graycomatrix(gray_scaled, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        
        # Color features (mean & std for each channel)
        mean_color = np.mean(img, axis=(0, 1))
        std_color = np.std(img, axis=(0, 1))

        # Flatten into a single feature vector
        feature_vector = np.hstack([contrast, dissimilarity, homogeneity, energy, correlation, mean_color, std_color])
        
        return feature_vector

    except Exception as e:
        print(f"[ERROR] Failed to extract features from {image_path}: {e}")
        return None
