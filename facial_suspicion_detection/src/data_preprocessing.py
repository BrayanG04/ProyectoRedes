import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import albumentations as A
"from imblearn.over_sampling import SMOTE"
import pandas as pd

class DataPreprocessor:
    def __init__(self, img_size=(128, 128)):
        self.img_size = img_size
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
    def detect_faces(self, image):
        """Detecta rostros en una imagen"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        return faces
    
    def preprocess_image(self, image_path):
        """Preprocesa una imagen individual"""
        try:
            # Leer imagen
            image = cv2.imread(image_path)
            if image is None:
                return None
                
            # Detectar rostros
            faces = self.detect_faces(image)
            if len(faces) == 0:
                return None
                
            # Tomar el primer rostro detectado
            x, y, w, h = faces[0]
            face = image[y:y+h, x:x+w]
            
            # Redimensionar y normalizar
            face = cv2.resize(face, self.img_size)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = face.astype(np.float32) / 255.0
            
            return face
            
        except Exception as e:
            print(f"Error procesando {image_path}: {e}")
            return None
    
    def augment_data(self, images, labels):
        """Aumenta los datos para balancear clases"""
        augmentations = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
        ])
        
        augmented_images = []
        augmented_labels = []
        
        for image, label in zip(images, labels):
            augmented = augmentations(image=image)
            augmented_images.append(augmented['image'])
            augmented_labels.append(label)
            
        return np.array(augmented_images), np.array(augmented_labels)
    
    def load_dataset(self, data_dir, test_size=0.2, validation_size=0.2):
        """Carga y divide el dataset"""
        images = []
        labels = []
        
        # Asumiendo estructura: data_dir/{suspicious, non_suspicious}/
        for label, category in enumerate(['non_suspicious', 'suspicious']):
            category_path = os.path.join(data_dir, category)
            
            if not os.path.exists(category_path):
                continue
                
            for img_file in os.listdir(category_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(category_path, img_file)
                    processed_img = self.preprocess_image(img_path)
                    
                    if processed_img is not None:
                        images.append(processed_img)
                        labels.append(label)
        
        images = np.array(images)
        labels = np.array(labels)
        
        # Dividir datos
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        return (X_train, X_val, X_test, y_train, y_val, y_test)