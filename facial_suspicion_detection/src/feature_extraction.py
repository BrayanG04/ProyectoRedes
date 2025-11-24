import cv2
import numpy as np
from skimage import feature
from tensorflow.keras.applications import VGG16 # type: ignore
from tensorflow.keras.models import Model # type: ignore
import tensorflow as tf

class FeatureExtractor:
    def __init__(self):
        self.lbp_params = {
            'radius': 2,
            'n_points': 16,
            'method': 'uniform'
        }
    
    def extract_lbp_features(self, image):
        """Extrae características LBP de una imagen"""
        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calcular LBP
        lbp = feature.local_binary_pattern(
            gray, 
            self.lbp_params['n_points'],
            self.lbp_params['radius'],
            self.lbp_params['method']
        )
        
        # Calcular histograma
        n_bins = self.lbp_params['n_points'] + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        
        # Normalizar
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        
        return hist
    
    def extract_deep_features(self, images, model_type='vgg16'):
        """Extrae características profundas usando CNN pre-entrenada"""
        if model_type == 'vgg16':
            base_model = VGG16(weights='imagenet', include_top=False, 
                              input_shape=(128, 128, 3))
            model = Model(inputs=base_model.input, 
                         outputs=base_model.output)
        
        # Preprocesar imágenes para el modelo
        if model_type == 'vgg16':
            preprocessed = tf.keras.applications.vgg16.preprocess_input(
                images * 255.0  # Desnormalizar
            )
        
        # Extraer características
        features = model.predict(preprocessed, verbose=0)
        features_flat = features.reshape(features.shape[0], -1)
        
        return features_flat
    
    def extract_hybrid_features(self, images):
        """Combina características LBP y profundas"""
        lbp_features = []
        for image in images:
            lbp_feat = self.extract_lbp_features(image)
            lbp_features.append(lbp_feat)
        
        lbp_features = np.array(lbp_features)
        deep_features = self.extract_deep_features(images)
        
        # Combinar características
        hybrid_features = np.concatenate([lbp_features, deep_features], axis=1)
        
        return hybrid_features