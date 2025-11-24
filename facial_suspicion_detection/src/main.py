#!/usr/bin/env python3
"""
Sistema de Predicción de Comportamiento Sospechoso mediante Expresiones Faciales
Autor: Brayan Camilo Gomez Rodríguez
Universidad Distrital Francisco Jose de Caldas
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime

# Añadir el directorio src al path
sys.path.append('src')

# Importar módulos personalizados
from data_preprocessing import DataPreprocessor
from feature_extraction import FeatureExtractor
from model_training import ModelTrainer
from evaluation import ModelEvaluator
from utils import Logger, VisualizationUtils, DataUtils, EthicalUtils

class FacialSuspicionDetection:
    """Clase principal del sistema de detección de comportamiento sospechoso"""
    
    def __init__(self, config=None):
        self.config = config or self.get_default_config()
        self.logger = Logger(self.config['log_dir'])
        self.preprocessor = None
        self.feature_extractor = None
        self.model_trainer = None
        self.evaluator = None
        
        # Resultados
        self.results = {}
        self.models = {}
        
    @staticmethod
    def get_default_config():
        """Retorna la configuración por defecto"""
        return {
            'data_dir': 'data/raw',
            'model_dir': 'data/models',
            'results_dir': 'results',
            'log_dir': 'logs',
            'img_size': (128, 128),
            'test_size': 0.2,
            'validation_size': 0.2,
            'epochs': 50,
            'batch_size': 32,
            'feature_type': 'hybrid',  # 'raw', 'lbp', 'hybrid'
            'use_data_augmentation': True,
            'train_cnn': True,
            'train_classical_models': True,
            'evaluate_bias': True,
            'random_seed': 42
        }
    
    def setup_environment(self):
        """Configura el entorno del proyecto"""
        self.logger.log("Configurando entorno del proyecto...")
        
        # Crear directorios necesarios
        for directory in [self.config['data_dir'], self.config['model_dir'], 
                         self.config['results_dir']]:
            os.makedirs(directory, exist_ok=True)
            
        # Establecer semillas para reproducibilidad
        np.random.seed(self.config['random_seed'])
        
        # Guardar configuración
        self.logger.save_experiment_config(self.config)
        self.logger.log("Entorno configurado correctamente")
    
    def load_and_preprocess_data(self):
        """Carga y preprocesa los datos"""
        self.logger.log("Cargando y preprocesando datos...")
        
        # Verificar calidad del dataset
        data_report = DataUtils.check_dataset_quality(self.config['data_dir'])
        self.logger.log(f"Reporte de calidad de datos: {data_report}")
        
        # Inicializar preprocesador
        self.preprocessor = DataPreprocessor(img_size=self.config['img_size'])
        
        # Cargar dataset
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.load_dataset(
            self.config['data_dir'],
            test_size=self.config['test_size'],
            validation_size=self.config['validation_size']
        )
        
        # Verificar que hay datos suficientes
        if len(X_train) == 0:
            raise ValueError("No se encontraron datos para entrenamiento")
        
        self.logger.log(f"Datos de entrenamiento: {X_train.shape}")
        self.logger.log(f"Datos de validación: {X_val.shape}")
        self.logger.log(f"Datos de prueba: {X_test.shape}")
        self.logger.log(f"Distribución entrenamiento: {np.bincount(y_train)}")
        self.logger.log(f"Distribución prueba: {np.bincount(y_test)}")
        
        # Aumentar datos si es necesario
        if self.config['use_data_augmentation'] and len(X_train) < 1000:
            self.logger.log("Aplicando aumento de datos...")
            X_train_aug, y_train_aug = self.preprocessor.augment_data(X_train, y_train)
            X_train = np.concatenate([X_train, X_train_aug])
            y_train = np.concatenate([y_train, y_train_aug])
            self.logger.log(f"Datos después de aumento: {X_train.shape}")
        
        # Visualizar datos
        VisualizationUtils.plot_sample_images(
            X_train, y_train, 
            ['No Sospechoso', 'Sospechoso'], 
            n_samples=8
        )
        VisualizationUtils.plot_class_distribution(
            y_train, 
            ['No Sospechoso', 'Sospechoso'], 
            "Datos de Entrenamiento"
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def extract_features(self, X_train, X_val, X_test):
        """Extrae características de las imágenes"""
        self.logger.log("Extrayendo características...")
        
        self.feature_extractor = FeatureExtractor()
        
        if self.config['feature_type'] == 'raw':
            # Usar imágenes raw para CNN
            X_train_features = X_train
            X_val_features = X_val
            X_test_features = X_test
            
        elif self.config['feature_type'] == 'lbp':
            # Extraer características LBP
            X_train_features = np.array([self.feature_extractor.extract_lbp_features(img) 
                                       for img in X_train])
            X_val_features = np.array([self.feature_extractor.extract_lbp_features(img) 
                                     for img in X_val])
            X_test_features = np.array([self.feature_extractor.extract_lbp_features(img) 
                                      for img in X_test])
            
        elif self.config['feature_type'] == 'hybrid':
            # Extraer características híbridas (LBP + Deep Features)
            self.logger.log("Extrayendo características híbridas...")
            X_train_features = self.feature_extractor.extract_hybrid_features(X_train)
            X_val_features = self.feature_extractor.extract_hybrid_features(X_val)
            X_test_features = self.feature_extractor.extract_hybrid_features(X_test)
        
        self.logger.log(f"Características de entrenamiento: {X_train_features.shape}")
        return X_train_features, X_val_features, X_test_features
    
    def train_models(self, X_train, y_train, X_val, y_val, X_train_features=None):
        """Entrena los modelos"""
        self.logger.log("Iniciando entrenamiento de modelos...")
        
        self.model_trainer = ModelTrainer()
        self.models = {}
        
        # Entrenar CNN
        if self.config['train_cnn'] and self.config['feature_type'] in ['raw', 'hybrid']:
            self.logger.log("Entrenando modelo CNN...")
            history = self.model_trainer.train_cnn(
                X_train, y_train, X_val, y_val,
                epochs=self.config['epochs'],
                batch_size=self.config['batch_size']
            )
            self.models['cnn'] = self.model_trainer.models['cnn']
            
            # Visualizar historial de entrenamiento
            VisualizationUtils.plot_training_history(history)
        
        # Entrenar modelos clásicos
        if self.config['train_classical_models'] and X_train_features is not None:
            self.logger.log("Entrenando modelos clásicos...")
            self.model_trainer.train_classical_models(
                X_train, y_train, 
                feature_type=self.config['feature_type']
            )
            
            # Agregar modelos clásicos al diccionario
            for name, model in self.model_trainer.models.items():
                if name != 'cnn':
                    self.models[name] = model
        
        self.logger.log(f"Modelos entrenados: {list(self.models.keys())}")
        return self.models
    
    def evaluate_models(self, X_test, y_test, X_test_features=None):
        """Evalúa todos los modelos"""
        self.logger.log("Evaluando modelos...")
        
        self.evaluator = ModelEvaluator(self.models, self.feature_extractor)
        
        # Evaluar modelos
        results = self.evaluator.evaluate_models(
            X_test, y_test, 
            feature_type=self.config['feature_type']
        )
        
        # Comparar modelos
        VisualizationUtils.plot_model_comparison(results)
        
        # Análisis de sesgos (si hay datos demográficos disponibles)
        if self.config['evaluate_bias']:
            self.logger.log("Realizando análisis de sesgos...")
            
            # Aquí podrías cargar datos demográficos reales
            # Por ahora, usamos un ejemplo simulado
            demographic_groups = np.random.choice([0, 1], size=len(y_test))
            group_names = ['Grupo_A', 'Grupo_B']
            
            for model_name, model_results in results.items():
                y_pred = model_results['predictions']
                
                bias_report = EthicalUtils.demographic_bias_report(
                    y_test, y_pred, demographic_groups, group_names
                )
                
                self.logger.log(f"Reporte de sesgos - {model_name}: {bias_report}")
                
                # Visualizar sesgos
                EthicalUtils.plot_bias_analysis(bias_report)
        
        self.results = results
        return results
    
    def save_results(self):
        """Guarda los resultados y modelos"""
        self.logger.log("Guardando resultados...")
        
        # Guardar modelos
        if self.model_trainer:
            self.model_trainer.save_models(self.config['model_dir'])
        
        # Guardar resultados
        if self.results:
            self.logger.save_results(self.results, "model_results.json")
        
        # Guardar análisis de características
        if self.feature_extractor and hasattr(self.feature_extractor, 'lbp_params'):
            feature_info = {
                'lbp_params': self.feature_extractor.lbp_params,
                'feature_type': self.config['feature_type'],
                'timestamp': datetime.now().isoformat()
            }
            self.logger.save_results(feature_info, "feature_info.json")
        
        self.logger.log("Resultados guardados correctamente")
    
    def run_complete_pipeline(self):
        """Ejecuta el pipeline completo del proyecto"""
        try:
            self.logger.log("=== INICIANDO PIPELINE COMPLETO ===")
            
            # 1. Configuración
            self.setup_environment()
            
            # 2. Carga y preprocesamiento de datos
            X_train, X_val, X_test, y_train, y_val, y_test = self.load_and_preprocess_data()
            
            # 3. Extracción de características
            X_train_features, X_val_features, X_test_features = self.extract_features(
                X_train, X_val, X_test
            )
            
            # 4. Entrenamiento de modelos
            models = self.train_models(
                X_train, y_train, X_val, y_val, X_train_features
            )
            
            # 5. Evaluación
            results = self.evaluate_models(
                X_test, y_test, X_test_features
            )
            
            # 6. Guardar resultados
            self.save_results()
            
            self.logger.log("=== PIPELINE COMPLETADO EXITOSAMENTE ===")
            
            # Mostrar resumen final
            self.print_final_summary(results)
            
            return results
            
        except Exception as e:
            self.logger.log(f"Error en el pipeline: {str(e)}", level="ERROR")
            raise
    
    def print_final_summary(self, results):
        """Imprime un resumen final de los resultados"""
        print("\n" + "="*60)
        print("RESUMEN FINAL DEL PROYECTO")
        print("="*60)
        
        for model_name, model_results in results.items():
            print(f"\n--- {model_name.upper()} ---")
            print(f"Precisión: {model_results['accuracy']:.4f}")
            print(f"Precisión: {model_results['precision']:.4f}")
            print(f"Recall: {model_results['recall']:.4f}")
            print(f"AUC-ROC: {model_results['roc_auc']:.4f}")
        
        print(f"\nMejor modelo: {max(results.items(), key=lambda x: x[1]['accuracy'])[0]}")
        print("="*60)

def parse_arguments():
    """Parsea los argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description='Sistema de Detección de Comportamiento Sospechoso')
    
    parser.add_argument('--data_dir', type=str, default='data/raw',
                       help='Directorio con los datos de entrenamiento')
    parser.add_argument('--model_dir', type=str, default='data/models',
                       help='Directorio para guardar modelos')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directorio para guardar resultados')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Número de épocas para entrenamiento')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Tamaño del batch')
    parser.add_argument('--feature_type', choices=['raw', 'lbp', 'hybrid'], 
                       default='hybrid', help='Tipo de características a extraer')
    parser.add_argument('--img_size', type=int, nargs=2, default=[128, 128],
                       help='Tamaño de las imágenes (ancho alto)')
    parser.add_argument('--no_cnn', action='store_true',
                       help='No entrenar modelo CNN')
    parser.add_argument('--no_classical', action='store_true',
                       help='No entrenar modelos clásicos')
    parser.add_argument('--no_bias_analysis', action='store_true',
                       help='No realizar análisis de sesgos')
    
    return parser.parse_args()

def main():
    """Función principal"""
    print("Sistema de Predicción de Comportamiento Sospechoso")
    print("Universidad Distrital Francisco Jose de Caldas")
    print("=" * 60)
    
    # Parsear argumentos
    args = parse_arguments()
    
    # Crear configuración
    config = FacialSuspicionDetection.get_default_config()
    
    # Actualizar configuración con argumentos
    config.update({
        'data_dir': args.data_dir,
        'model_dir': args.model_dir,
        'results_dir': args.results_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'feature_type': args.feature_type,
        'img_size': tuple(args.img_size),
        'train_cnn': not args.no_cnn,
        'train_classical_models': not args.no_classical,
        'evaluate_bias': not args.no_bias_analysis
    })
    
    # Crear y ejecutar el sistema
    system = FacialSuspicionDetection(config)
    
    try:
        results = system.run_complete_pipeline()
        print("\n✅ Proyecto ejecutado exitosamente!")
        return results
        
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
        return None

if __name__ == "__main__":
    main()