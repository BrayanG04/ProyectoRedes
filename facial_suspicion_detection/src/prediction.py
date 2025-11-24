import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import os
import sys
from datetime import datetime

class FacialSuspicionPredictor:
    """Clase para hacer predicciones con modelos entrenados - VERSIÓN SIMPLIFICADA"""
    
    def __init__(self, model_dir="data/models", img_size=(128, 128)):
        self.model_dir = model_dir
        self.img_size = img_size
        self.models = {}
        self.feature_extractor = None
        self.analysis_history = []
        
        # Umbral ajustable
        self.threshold = 0.3
        
        # Cargar detector de rostros
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def set_threshold(self, threshold):
        """Permite ajustar el umbral de decisión"""
        if 0 <= threshold <= 1:
            self.threshold = threshold
            print(f"🎚️  Umbral ajustado a: {threshold}")
            return True
        else:
            print("❌ Umbral debe estar entre 0 y 1")
            return False
    
    def load_models(self):
        """Carga solo CNN y Random Forest"""
        try:
            models_loaded = 0
            model_files = {
                'cnn': 'cnn_model.h5',
                'random_forest': 'random_forest_model.joblib'
            }
            
            print("🔧 Cargando modelos...")
            for model_name, model_file in model_files.items():
                model_path = os.path.join(self.model_dir, model_file)
                
                if os.path.exists(model_path):
                    try:
                        if model_name == 'cnn':
                            self.models[model_name] = load_model(model_path)
                            print(f"   ✅ CNN cargado")
                        else:
                            self.models[model_name] = joblib.load(model_path)
                            print(f"   ✅ Random Forest cargado")
                        models_loaded += 1
                    except Exception as e:
                        print(f"   ❌ Error cargando {model_name}: {e}")
                else:
                    print(f"   ⚠️  No encontrado: {model_file}")
            
            if models_loaded == 0:
                print("❌ No se pudieron cargar modelos")
                return False
            
            print(f"📦 {models_loaded} modelos cargados exitosamente")
            return True
            
        except Exception as e:
            print(f"❌ Error crítico cargando modelos: {e}")
            return False
    
    def preprocess_single_image(self, image_path):
        """Preprocesa una imagen individual"""
        try:
            if not os.path.exists(image_path):
                raise ValueError(f"Archivo no existe: {image_path}")
            
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"No se pudo decodificar la imagen: {image_path}")
            
            original_size = image.shape[:2]
            
            # Convertir a RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detección de rostros
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            detection_params = [
                {'scaleFactor': 1.1, 'minNeighbors': 5, 'minSize': (30, 30)},
                {'scaleFactor': 1.05, 'minNeighbors': 3, 'minSize': (40, 40)}
            ]
            
            faces = []
            for params in detection_params:
                faces = self.face_cascade.detectMultiScale(gray, **params)
                if len(faces) > 0:
                    break
            
            if len(faces) == 0:
                face = image_rgb
                face_coords = (0, 0, image.shape[1], image.shape[0])
                detection_quality = "baja"
            else:
                faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                x, y, w, h = faces[0]
                face = image_rgb[y:y+h, x:x+w]
                face_coords = (x, y, w, h)
                
                face_ratio = (w * h) / (original_size[0] * original_size[1])
                if face_ratio > 0.3:
                    detection_quality = "alta"
                elif face_ratio > 0.15:
                    detection_quality = "media"
                else:
                    detection_quality = "baja"
            
            # Preprocesamiento
            face = cv2.resize(face, self.img_size)
            
            if len(face.shape) == 3:
                lab = cv2.cvtColor(face, cv2.COLOR_RGB2LAB)
                lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0).apply(lab[:,:,0])
                face = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                face = cv2.GaussianBlur(face, (3, 3), 0)
            
            face = face.astype(np.float32) / 255.0
            
            return face, face_coords, detection_quality
            
        except Exception as e:
            print(f"❌ Error preprocesando imagen: {e}")
            return None, None, "error"
    
    def extract_features(self, image, feature_type='hybrid'):
        """Extrae características de una imagen"""
        try:
            from feature_extraction import FeatureExtractor
            
            if self.feature_extractor is None:
                self.feature_extractor = FeatureExtractor()
            
            if feature_type == 'lbp':
                return self.feature_extractor.extract_lbp_features(image)
            elif feature_type == 'hybrid':
                images_batch = np.expand_dims(image, axis=0)
                features = self.feature_extractor.extract_hybrid_features(images_batch)
                return features[0]
            else:
                return image.flatten()
                
        except Exception as e:
            print(f"❌ Error extrayendo características: {e}")
            return None
    
    def predict_single_image(self, image_path, model_type='cnn'):
        """
        Predice si una imagen es sospechosa - VERSIÓN SIMPLIFICADA
        """
        print(f"\n🎯 ANALIZANDO IMAGEN")
        print(f"   📁 Imagen: {os.path.basename(image_path)}")
        print(f"   🤖 Modelo: {model_type}")
        
        if model_type not in self.models:
            error_msg = f"Modelo {model_type} no disponible"
            print(f"❌ {error_msg}")
            return {'success': False, 'error': error_msg}
        
        processed_image, face_coords, detection_quality = self.preprocess_single_image(image_path)
        if processed_image is None:
            return {'success': False, 'error': 'No se pudo procesar la imagen'}
        
        try:
            # Realizar predicción
            if model_type == 'cnn':
                input_image = np.expand_dims(processed_image, axis=0)
                prediction_prob = self.models['cnn'].predict(input_image, verbose=0)[0][0]
            else:
                features = self.extract_features(processed_image, feature_type='hybrid')
                if features is None:
                    return {'success': False, 'error': 'Error extrayendo características'}
                    
                features = np.expand_dims(features, axis=0)
                prediction_prob = self.models[model_type].predict_proba(features)[0][1]
            
            # Interpretar resultado
            is_suspicious = prediction_prob > self.threshold
            confidence = prediction_prob if is_suspicious else 1 - prediction_prob
            class_label = "SOSPECHOSO" if is_suspicious else "NO SOSPECHOSO"
            
            # Resultado simplificado
            result = {
                'success': True,
                'class': class_label,
                'is_suspicious': bool(is_suspicious),
                'confidence': float(confidence),
                'probability': float(prediction_prob),
                'face_coordinates': face_coords,
                'model_used': model_type,
                'threshold_used': self.threshold,
                'detection_quality': detection_quality,
                'timestamp': datetime.now().isoformat()
            }
            
            # Guardar en historial
            self.analysis_history.append(result)
            
            # Mostrar resultado simple
            self.print_simple_result(result)
            
            return result
            
        except Exception as e:
            error_msg = f'Error en predicción: {str(e)}'
            print(f"❌ {error_msg}")
            return {'success': False, 'error': error_msg}
    
    def print_simple_result(self, result):
        """Muestra solo el resultado básico"""
        print(f"\n📊 RESULTADO")
        print("=" * 30)
        print(f"🎯 CLASIFICACIÓN: {result['class']}")
        print(f"🔍 CALIDAD DETECCIÓN: {result['detection_quality']}")
        print("=" * 30)
    
    def compare_models(self, image_path):
        """Compara CNN y Random Forest para una imagen"""
        print(f"\n🔍 COMPARANDO MODELOS")
        print("=" * 40)
        
        results = {}
        for model_name in ['cnn', 'random_forest']:
            if model_name in self.models:
                print(f"\n🤖 Probando {model_name}...")
                result = self.predict_single_image(image_path, model_name)
                if result['success']:
                    results[model_name] = result
                    print(f"   ✅ {result['class']}")
                else:
                    print(f"   ❌ Error: {result['error']}")
        
        # Análisis comparativo simple
        if results:
            print(f"\n📈 COMPARACIÓN")
            print("=" * 20)
            
            classifications = [r['class'] for r in results.values()]
            if all(c == classifications[0] for c in classifications):
                print("✅ Los modelos coinciden")
            else:
                print("⚠️  Los modelos no coinciden")
        
        return results
    
    def get_analysis_history(self):
        """Obtiene el historial de análisis"""
        return self.analysis_history[-10:]
    
    def predict_batch(self, image_paths, model_type='cnn'):
        """Predice múltiples imágenes"""
        print(f"\n🔄 ANALIZANDO {len(image_paths)} IMÁGENES")
        results = []
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"\n📦 {i}/{len(image_paths)}: {os.path.basename(image_path)}")
            try:
                result = self.predict_single_image(image_path, model_type)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                print(f"❌ Error: {e}")
                results.append({
                    'success': False,
                    'image_path': image_path,
                    'error': str(e)
                })
        
        # Estadísticas simples
        successful = [r for r in results if r['success']]
        if successful:
            suspicious_count = sum(1 for r in successful if r['is_suspicious'])
            
            print(f"\n📊 ESTADÍSTICAS")
            print(f"   ✅ Éxitos: {len(successful)}/{len(image_paths)}")
            print(f"   🔴 Sospechosos: {suspicious_count}")
            print(f"   🟢 No sospechosos: {len(successful) - suspicious_count}")
        
        return results

    def visualize_prediction(self, image_path, prediction_result, save_path=None):
        """Visualiza la predicción en la imagen - VERSIÓN SIMPLIFICADA"""
        import matplotlib.pyplot as plt
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(10, 8))
        
        # Dibujar rectángulo si se detectó rostro
        if prediction_result['face_coordinates'] and prediction_result['face_coordinates'] != (0, 0, image.shape[1], image.shape[0]):
            x, y, w, h = prediction_result['face_coordinates']
            color = (255, 0, 0) if prediction_result['is_suspicious'] else (0, 255, 0)
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 3)
            
            label = f"{prediction_result['class']}"
            cv2.putText(image, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        plt.imshow(image)
        plt.title(f"Análisis Facial - {prediction_result['class']}")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"💾 Visualización guardada")
        
        plt.tight_layout()
        plt.show()

# Script de uso rápido simplificado
def main():
    """Ejemplo de uso del predictor simplificado"""
    predictor = FacialSuspicionPredictor()
    
    # Cargar modelos
    if not predictor.load_models():
        print("❌ No se pudieron cargar los modelos")
        return
    
    print(f"\n🎯 PREDICTOR INICIALIZADO")
    print(f"📊 Modelos disponibles: {list(predictor.models.keys())}")
    
    while True:
        print("\n" + "="*40)
        print("1. Analizar imagen")
        print("2. Comparar modelos")
        print("3. Cambiar umbral")
        print("4. Salir")
        
        option = input("\nSelecciona opción: ").strip()
        
        if option == "1":
            image_path = input("Ruta de la imagen: ").strip()
            if os.path.exists(image_path):
                model_choice = input("Modelo (Enter para CNN): ").strip() or "cnn"
                if model_choice in predictor.models:
                    result = predictor.predict_single_image(image_path, model_choice)
                    if result['success']:
                        predictor.visualize_prediction(image_path, result)
                else:
                    print("❌ Modelo no válido")
            else:
                print("❌ La imagen no existe")
                
        elif option == "2":
            image_path = input("Ruta de la imagen: ").strip()
            if os.path.exists(image_path):
                predictor.compare_models(image_path)
            else:
                print("❌ La imagen no existe")
                
        elif option == "3":
            try:
                new_threshold = float(input("Nuevo umbral (0-1): "))
                predictor.set_threshold(new_threshold)
            except ValueError:
                print("❌ Umbral debe ser un número")
                
        elif option == "4":
            print("👋 ¡Hasta luego!")
            break
        else:
            print("❌ Opción no válida")

if __name__ == "__main__":
    main()