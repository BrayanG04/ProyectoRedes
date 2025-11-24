import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import joblib

class ModelTrainer:
    def __init__(self):
        self.models = {}
        
    def create_cnn_model(self, input_shape=(128, 128, 3)):
        """Crea un modelo CNN personalizado - VERSIÓN SIMPLIFICADA"""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Flatten(),  # Cambiado de GlobalAveragePooling2D para más estabilidad
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        # ✅ SOLO accuracy - Las otras métricas se calcularán manualmente
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']  # Solo accuracy para evitar errores
        )
        
        return model
    
    def train_cnn(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Entrena el modelo CNN"""
        model = self.create_cnn_model()
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7
            )
        ]
        
        print("🔄 Entrenando CNN...")
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        self.models['cnn'] = model
        return history
    
    def train_classical_models(self, X_train, y_train, feature_type='hybrid'):
        """Entrena modelos clásicos de ML"""
        if feature_type == 'lbp':
            from feature_extraction import FeatureExtractor
            feature_extractor = FeatureExtractor()
            X_train_features = np.array([feature_extractor.extract_lbp_features(img) 
                                       for img in X_train])
        elif feature_type == 'hybrid':
            from feature_extraction import FeatureExtractor
            feature_extractor = FeatureExtractor()
            X_train_features = feature_extractor.extract_hybrid_features(X_train)
        else:
            X_train_features = X_train.reshape(X_train.shape[0], -1)
        
        print("🌲 Entrenando Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        rf_model.fit(X_train_features, y_train)
        self.models['random_forest'] = rf_model
        
        print("📊 Entrenando SVM...")
        svm_model = SVC(
            kernel='rbf',
            C=1.0,
            probability=True,
            random_state=42,
            class_weight='balanced'
        )
        svm_model.fit(X_train_features, y_train)
        self.models['svm'] = svm_model
        
        print("📈 Entrenando Regresión Logística...")
        lr_model = LogisticRegression(
            C=1.0,
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        )
        lr_model.fit(X_train_features, y_train)
        self.models['logistic_regression'] = lr_model
        
        return X_train_features
    
    def save_models(self, model_dir):
        """Guarda los modelos entrenados"""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        for name, model in self.models.items():
            if name == 'cnn':
                model.save(os.path.join(model_dir, f'{name}_model.h5'))
                print(f"💾 CNN guardado: {name}_model.h5")
            else:
                joblib.dump(model, os.path.join(model_dir, f'{name}_model.joblib'))
                print(f"💾 {name} guardado: {name}_model.joblib")