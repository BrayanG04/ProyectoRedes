import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import cv2
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

class Logger:
    """Clase para logging y seguimiento de experimentos"""
    
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Crear archivo de log
        self.log_file = os.path.join(
            log_dir, 
            f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
    
    def log(self, message, level="INFO"):
        """Registra un mensaje en el log"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] [{level}] {message}"
        
        print(log_message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def save_experiment_config(self, config):
        """Guarda la configuración del experimento"""
        config_file = os.path.join(self.log_dir, "experiment_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
    
    def save_results(self, results, filename="results.json"):
        """Guarda los resultados del experimento"""
        results_file = os.path.join(self.log_dir, filename)
        
        # Convertir numpy arrays a listas para serialización JSON
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, (np.ndarray, np.generic)):
                serializable_results[key] = value.tolist()
            elif isinstance(value, dict):
                serializable_results[key] = {
                    k: v.tolist() if isinstance(v, (np.ndarray, np.generic)) else v
                    for k, v in value.items()
                }
            else:
                serializable_results[key] = value
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=4, ensure_ascii=False)

class VisualizationUtils:
    """Utilidades para visualización de datos y resultados"""
    
    @staticmethod
    def plot_sample_images(images, labels, class_names, n_samples=10):
        """Muestra una muestra de imágenes del dataset"""
        n_classes = len(class_names)
        fig, axes = plt.subplots(2, n_classes, figsize=(15, 6))
        
        for class_idx in range(n_classes):
            # Seleccionar imágenes de la clase actual
            class_images = images[labels == class_idx]
            
            if len(class_images) > 0:
                # Mostrar primeras imágenes
                for i in range(min(2, len(class_images))):
                    ax = axes[i, class_idx]
                    ax.imshow(class_images[i])
                    ax.set_title(f'{class_names[class_idx]} - Ejemplo {i+1}')
                    ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_class_distribution(labels, class_names, dataset_name=""):
        """Grafica la distribución de clases"""
        class_counts = np.bincount(labels)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(class_names, class_counts, color=['lightblue', 'lightcoral'])
        plt.title(f'Distribución de Clases - {dataset_name}')
        plt.xlabel('Clases')
        plt.ylabel('Número de Imágenes')
        
        # Añadir valores en las barras
        for bar, count in zip(bars, class_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{count}', ha='center', va='bottom')
        
        plt.show()
    
    @staticmethod
    def plot_training_history(history):
        """Grafica el historial de entrenamiento del modelo"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Gráfico de pérdida
        axes[0].plot(history.history['loss'], label='Pérdida Entrenamiento')
        axes[0].plot(history.history['val_loss'], label='Pérdida Validación')
        axes[0].set_title('Evolución de la Pérdida')
        axes[0].set_xlabel('Época')
        axes[0].set_ylabel('Pérdida')
        axes[0].legend()
        axes[0].grid(True)
        
        # Gráfico de precisión
        axes[1].plot(history.history['accuracy'], label='Precisión Entrenamiento')
        axes[1].plot(history.history['val_accuracy'], label='Precisión Validación')
        axes[1].set_title('Evolución de la Precisión')
        axes[1].set_xlabel('Época')
        axes[1].set_ylabel('Precisión')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_feature_importance(feature_names, importance_scores, top_k=20):
        """Grafica la importancia de las características (para modelos como Random Forest)"""
        # Combinar nombres y scores
        feature_importance = list(zip(feature_names, importance_scores))
        
        # Ordenar por importancia
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Tomar las top_k características
        top_features = feature_importance[:top_k]
        
        # Crear gráfico
        plt.figure(figsize=(12, 8))
        features, importances = zip(*top_features)
        
        y_pos = np.arange(len(features))
        plt.barh(y_pos, importances, align='center')
        plt.yticks(y_pos, features)
        plt.xlabel('Importancia')
        plt.title(f'Top {top_k} Características Más Importantes')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_model_comparison(results):
        """Compara el rendimiento de diferentes modelos"""
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'roc_auc']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in models]
            bars = axes[i].bar(models, values, color=plt.cm.Set3(np.linspace(0, 1, len(models))))
            axes[i].set_title(f'Comparación de {metric.upper()}')
            axes[i].set_ylabel(metric.upper())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Añadir valores en las barras
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

class DataUtils:
    """Utilidades para manejo y verificación de datos"""
    
    @staticmethod
    def check_dataset_quality(data_dir, expected_classes=['suspicious', 'non_suspicious']):
        """Verifica la calidad y completitud del dataset"""
        report = {
            'total_images': 0,
            'class_distribution': {},
            'image_formats': {},
            'potential_issues': []
        }
        
        for class_name in expected_classes:
            class_path = os.path.join(data_dir, class_name)
            
            if not os.path.exists(class_path):
                report['potential_issues'].append(f'Directorio de clase faltante: {class_name}')
                report['class_distribution'][class_name] = 0
                continue
            
            # Contar imágenes por formato
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            report['class_distribution'][class_name] = len(image_files)
            report['total_images'] += len(image_files)
            
            # Analizar formatos
            for img_file in image_files:
                ext = os.path.splitext(img_file)[1].lower()
                report['image_formats'][ext] = report['image_formats'].get(ext, 0) + 1
            
            # Verificar balance
            if len(image_files) == 0:
                report['potential_issues'].append(f'Clase sin imágenes: {class_name}')
        
        # Verificar balance de clases
        class_counts = list(report['class_distribution'].values())
        if len(class_counts) > 1 and max(class_counts) / min(class_counts) > 2:
            report['potential_issues'].append('Desbalance significativo entre clases')
        
        return report
    
    @staticmethod
    def analyze_image_characteristics(images):
        """Analiza características de las imágenes"""
        if len(images) == 0:
            return {}
        
        # Convertir a numpy array si es necesario
        images_array = np.array(images)
        
        analysis = {
            'shape': images_array.shape,
            'data_type': images_array.dtype,
            'min_value': float(np.min(images_array)),
            'max_value': float(np.max(images_array)),
            'mean_value': float(np.mean(images_array)),
            'std_value': float(np.std(images_array)),
            'memory_size_mb': images_array.nbytes / (1024 * 1024)
        }
        
        return analysis
    
    @staticmethod
    def save_processed_data(X, y, filepath):
        """Guarda datos procesados en formato numpy"""
        np.savez_compressed(filepath, X=X, y=y)
        print(f"Datos guardados en: {filepath}")
    
    @staticmethod
    def load_processed_data(filepath):
        """Carga datos procesados desde archivo numpy"""
        data = np.load(filepath)
        return data['X'], data['y']

class EthicalUtils:
    """Utilidades para análisis ético y de sesgos"""
    
    @staticmethod
    def demographic_bias_report(y_true, y_pred, demographic_groups, group_names):
        """Genera reporte de sesgos demográficos"""
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        bias_report = {}
        
        for i, group_name in enumerate(group_names):
            group_mask = demographic_groups == i
            
            if np.sum(group_mask) == 0:
                continue
            
            y_true_group = y_true[group_mask]
            y_pred_group = y_pred[group_mask]
            
            precision = precision_score(y_true_group, y_pred_group, zero_division=0)
            recall = recall_score(y_true_group, y_pred_group, zero_division=0)
            f1 = f1_score(y_true_group, y_pred_group, zero_division=0)
            accuracy = np.mean(y_true_group == y_pred_group)
            
            # Tasa de falsos positivos y negativos
            tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            bias_report[group_name] = {
                'n_samples': len(y_true_group),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'false_positive_rate': fpr,
                'false_negative_rate': fnr,
                'equalized_odds_diff': abs(fpr - fnr)
            }
        
        return bias_report
    
    @staticmethod
    def fairness_metrics(y_true, y_pred, demographic_groups):
        """Calcula métricas de fairness"""
        from sklearn.metrics import confusion_matrix
        
        unique_groups = np.unique(demographic_groups)
        metrics_by_group = {}
        
        for group in unique_groups:
            group_mask = demographic_groups == group
            y_true_group = y_true[group_mask]
            y_pred_group = y_pred[group_mask]
            
            tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
            
            metrics_by_group[group] = {
                'true_positive_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
                'positive_predictive_value': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0,
            }
        
        return metrics_by_group
    
    @staticmethod
    def plot_bias_analysis(bias_report):
        """Visualiza el análisis de sesgos"""
        groups = list(bias_report.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [bias_report[group][metric] for group in groups]
            bars = axes[i].bar(groups, values, color=plt.cm.Set3(np.linspace(0, 1, len(groups))))
            axes[i].set_title(f'{metric.upper()} por Grupo Demográfico')
            axes[i].set_ylabel(metric.upper())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Añadir valores en las barras
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

class ModelUtils:
    """Utilidades para manejo de modelos"""
    
    @staticmethod
    def calculate_model_size(model_path):
        """Calcula el tamaño del modelo en disco"""
        if os.path.exists(model_path):
            size_bytes = os.path.getsize(model_path)
            size_mb = size_bytes / (1024 * 1024)
            return size_mb
        return 0
    
    @staticmethod
    def load_model(model_path, model_type='keras'):
        """Carga un modelo guardado"""
        if model_type == 'keras':
            return tf.keras.models.load_model(model_path)
        else:
            import joblib
            return joblib.load(model_path)
    
    @staticmethod
    def model_summary_to_file(model, filepath):
        """Guarda el resumen del modelo en un archivo"""
        with open(filepath, 'w', encoding='utf-8') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

# Funciones de utilidad general
def create_directory_structure():
    """Crea la estructura de directorios del proyecto"""
    directories = [
        'data/raw/suspicious',
        'data/raw/non_suspicious',
        'data/processed',
        'data/models',
        'logs',
        'results',
        'notebooks',
        'src'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Directorio creado: {directory}")

def get_timestamp():
    """Retorna un timestamp formateado"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def set_random_seeds(seed=42):
    """Establece semillas para reproducibilidad"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def print_system_info():
    """Imprime información del sistema"""
    import platform
    import tensorflow as tf
    
    print("=== INFORMACIÓN DEL SISTEMA ===")
    print(f"Sistema Operativo: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"OpenCV: {cv2.__version__}")
    print("================================")