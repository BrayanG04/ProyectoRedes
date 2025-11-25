# 🎭 Facial Suspicion Detection System

Sistema de inteligencia artificial para predicción de comportamiento
sospechoso mediante análisis de expresiones faciales en imágenes
estáticas.

## 📋 Descripción

Este proyecto implementa un sistema completo de machine learning que
utiliza técnicas de visión por computadora y aprendizaje profundo para
analizar expresiones faciales y predecir comportamientos sospechosos. El
sistema combina redes neuronales convolucionales (CNN) con modelos
clásicos de machine learning para lograr alta precisión y robustez.

## 🚀 Características Principales

-   **🔍 Detección Facial Automática**: Usa OpenCV y Haar Cascades para
    detección robusta de rostros\
-   **🧠 Múltiples Modelos**: Implementa CNN, Random Forest, SVM y
    Regresión Logística\
-   **🎨 Interfaz Gráfica**: GUI intuitiva desarrollada con Tkinter\
-   **📊 Evaluación Comprehensiva**: Métricas múltiples y
    visualizaciones detalladas\
-   **⚖️ Análisis de Sesgos**: Herramientas para detectar y mitigar
    sesgos demográficos\
-   **🔄 Pipeline Completo**: Desde preprocesamiento hasta predicción en
    tiempo real

## 🏗️ Estructura del Proyecto

    FACIAL_SUSPICION_DETECTION/
    ├── data/
    │   ├── models/        # Modelos entrenados
    │   │   ├── cnn_model.h5
    │   │   ├── random_forest_model.joblib
    │   │   ├── svm_model.joblib
    │   │   └── logistic_regression_model.joblib
    │   └── raw/           # Dataset de entrenamiento
    │       ├── suspicious/
    │       └── non_suspicious/
    ├── logs/              # Registros de experimentos
    ├── results/           # Métricas y visualizaciones
    └── src/               # Código fuente
        ├── data_preprocessing.py
        ├── feature_extraction.py
        ├── model_training.py
        ├── evaluation.py
        ├── prediction.py
        ├── predict_image.py
        ├── main.py
        └── utils.py

## ⚙️ Instalación

### Prerrequisitos

-   Python 3.8 o superior\
-   pip

### Instalación de Dependencias

    # Clonar el repositorio
    git clone https://github.com/tu-usuario/facial-suspicion-detection.git
    cd facial-suspicion-detection

    # Instalar dependencias
    pip install -r src/requirements.txt

### Dependencias Principales

-   tensorflow\
-   opencv-python\
-   scikit-learn\
-   albumentations\
-   matplotlib, seaborn\
-   tkinter

## 🎯 Uso Rápido

### 1. Entrenamiento de Modelos

    python src/main.py --data_dir data/raw --epochs 50 --feature_type hybrid

### 2. Interfaz Gráfica

    python src/predict_image.py

### 3. Uso por Línea de Comandos

``` python
from src.prediction import FacialSuspicionPredictor

predictor = FacialSuspicionPredictor(model_dir="data/models")
predictor.load_models()

result = predictor.predict_single_image("ruta/a/imagen.jpg", model_type="cnn")
print(f"Clasificación: {result['class']}")
print(f"Confianza: {result['confidence']:.2%}")
```

## 📁 Estructura de Datos

    data/raw/
    ├── suspicious/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    └── non_suspicious/
        ├── img1.jpg
        ├── img2.jpg
        └── ...

Formatos soportados: JPG, PNG, BMP, TIFF.

## 🧩 Módulos Principales

### 1. Preprocesamiento

-   Detección facial\
-   Normalización\
-   Aumento de datos\
-   Balanceo de clases

### 2. Extracción de Características

-   LBP\
-   VGG16 pre-entrenada\
-   Híbrido

### 3. Entrenamiento

-   CNN\
-   Random Forest\
-   SVM\
-   Regresión Logística

### 4. Evaluación

-   Accuracy, Precision, Recall, F1, AUC-ROC\
-   Matriz de confusión\
-   Curvas ROC y PR\
-   Comparativa

### 5. Predicción

-   Procesamiento en tiempo real\
-   GUI interactiva\
-   Selección de modelo

## 📊 Métricas y Rendimiento

-   Precisión: \>85%\
-   Robustez ante cambios de iluminación\
-   Procesamiento rápido\
-   Flexible y escalable

## ⚠️ Consideraciones Éticas

-   Análisis de sesgos\
-   Interpretabilidad\
-   Transparencia\
-   Herramienta de apoyo, no decisiva

## 🐛 Solución de Problemas

**Error: "No se encontraron modelos"**\
Crear carpeta:

    mkdir -p data/models

**Error: "No se pudo decodificar la imagen"**\
- Revisar formato\
- Revisar ruta

**Error: Dependencias faltantes**

    pip install --upgrade -r src/requirements.txt
