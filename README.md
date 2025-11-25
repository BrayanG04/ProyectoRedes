# 🎭 Facial Suspicion Detection System

Sistema de inteligencia artificial para predicción de comportamiento sospechoso mediante análisis de expresiones faciales en imágenes estáticas.

## 📋 Descripción

Este proyecto implementa un sistema completo de machine learning que utiliza técnicas de visión por computadora y aprendizaje profundo para analizar expresiones faciales y predecir comportamientos sospechosos. El sistema combina redes neuronales convolucionales (CNN) con modelos clásicos de machine learning para lograr alta precisión y robustez.

## 🚀 Características Principales

- **🔍 Detección Facial Automática**: Usa OpenCV y Haar Cascades para detección robusta de rostros
- **🧠 Múltiples Modelos**: Implementa CNN, Random Forest, SVM y Regresión Logística
- **🎨 Interfaz Gráfica**: GUI intuitiva desarrollada con Tkinter
- **📊 Evaluación Comprehensiva**: Métricas múltiples y visualizaciones detalladas
- **⚖️ Análisis de Sesgos**: Herramientas para detectar y mitigar sesgos demográficos
- **🔄 Pipeline Completo**: Desde preprocesamiento hasta predicción en tiempo real

## 🏗️ Estructura del Proyecto
FACIAL_SUSPICION_DETECTION/
├── data/
│ ├── models/ # Modelos entrenados
│ │ ├── cnn_model.h5
│ │ ├── random_forest_model.joblib
│ │ ├── svm_model.joblib
│ │ └── logistic_regression_model.joblib
│ └── raw/ # Dataset de entrenamiento
│ ├── suspicious/
│ └── non_suspicious/
├── logs/ # Registros de experimentos
├── results/ # Métricas y visualizaciones
└── src/ # Código fuente
├── data_preprocessing.py
├── feature_extraction.py
├── model_training.py
├── evaluation.py
├── prediction.py
├── predict_image.py
├── main.py
└── utils.py

text

## ⚙️ Instalación

### Prerrequisitos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Instalación de Dependencias


# Clonar el repositorio
git clone https://github.com/tu-usuario/facial-suspicion-detection.git
cd facial-suspicion-detection

# Instalar dependencias
pip install -r src/requirements.txt
Dependencias Principales
tensorflow - Redes neuronales y deep learning

opencv-python - Procesamiento de imágenes y visión por computadora

scikit-learn - Machine learning tradicional

albumentations - Aumento de datos

matplotlib, seaborn - Visualizaciones

tkinter - Interfaz gráfica

🎯 Uso Rápido
1. Entrenamiento de Modelos
bash
# Ejecutar pipeline completo de entrenamiento
python src/main.py --data_dir data/raw --epochs 50 --feature_type hybrid
2. Interfaz Gráfica para Predicciones
bash
# Lanzar interfaz de usuario
python src/predict_image.py
3. Uso por Línea de Comandos
python
from src.prediction import FacialSuspicionPredictor

# Inicializar predictor
predictor = FacialSuspicionPredictor(model_dir="data/models")
predictor.load_models()

# Realizar predicción
result = predictor.predict_single_image("ruta/a/imagen.jpg", model_type="cnn")
print(f"Clasificación: {result['class']}")
print(f"Confianza: {result['confidence']:.2%}")
📁 Estructura de Datos
Dataset de Entrenamiento
Organiza tus imágenes en la siguiente estructura:

text
data/raw/
├── suspicious/         # Imágenes con comportamiento sospechoso
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── non_suspicious/    # Imágenes sin comportamiento sospechoso
    ├── img1.jpg
    ├── img2.jpg
    └── ...
Formatos Soportados
JPEG (.jpg, .jpeg)

PNG (.png)

BMP (.bmp)

TIFF (.tiff)

🧩 Módulos Principales
1. Preprocesamiento (data_preprocessing.py)
Detección facial automática

Normalización y redimensionamiento

Aumento de datos con Albumentations

Balanceo de clases

2. Extracción de Características (feature_extraction.py)
LBP (Local Binary Patterns): Texturas locales

Características profundas: VGG16 pre-entrenada

Características híbridas: Combinación optimizada

3. Entrenamiento (model_training.py)
CNN Personalizada: Arquitectura profunda para imágenes

Random Forest: Ensemble robusto para características

SVM y Regresión Logística: Modelos de comparación

4. Evaluación (evaluation.py)
Métricas: Accuracy, Precision, Recall, F1-Score, AUC-ROC

Matrices de confusión

Curvas ROC y Precision-Recall

Análisis comparativo

5. Predicción (prediction.py, predict_image.py)
Procesamiento en tiempo real

Interfaz gráfica intuitiva

Múltiples modelos seleccionables

Visualización de resultados

📊 Métricas y Rendimiento
El sistema ha demostrado:

Precisión: >85% en validación cruzada

Robustez: Manejo de variaciones en iluminación y pose

Velocidad: Procesamiento en segundos por imagen

Flexibilidad: Soporte para múltiples escenarios

⚠️ Consideraciones Éticas
Este sistema incluye herramientas para:

Detección de sesgos demográficos

Análisis de fairness entre grupos

Transparencia en las predicciones

Interpretabilidad de resultados

Importante: Este sistema debe usarse como herramienta de apoyo y no como único criterio para toma de decisiones.

🐛 Solución de Problemas
Error: "No se encontraron modelos"
bash
# Asegurar que los modelos estén en la ruta correcta
mkdir -p data/models
Error: "No se pudo decodificar la imagen"
Verificar que la imagen esté en formato soportado

Confirmar que la ruta sea correcta

Error: Dependencias faltantes
bash
# Reinstalar dependencias
