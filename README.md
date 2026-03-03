# 🍎 Máquina Separadora de Frutas Inteligente 🍊

Este proyecto tiene como objetivo el entrenamiento y despliegue de un modelo de **Machine Learning** capaz de identificar y clasificar distintas frutas mediante técnicas de **Visión Artificial**.

El sistema procesa imágenes, extrae características morfológicas, de color y textura, y utiliza un clasificador para determinar el tipo de fruta presente.

---

## 📂 Descripción de los Scripts

A continuación se detalla la función de cada script incluido en este repositorio:

### 1. 🧪 Extracción de Características
*   **`fruit_feature_extractor.py`**: Es el script principal para procesar el `dataset`. Segmenta las frutas usando el método de Otsu y extrae características como área, perímetro, circularidad, momentos de Hu y estadísticas de color HSV. Genera el archivo `fruit_features.csv`.
*   **`robust_feature_extractor.py`**: Una versión mejorada del extractor que utiliza el algoritmo **GrabCut** para una segmentación más precisa, especialmente en imágenes con fondos complejos. Genera `fruit_features_robust.csv`.

### 2. 🧠 Entrenamiento del Modelo
*   **`train_model.py`**: Carga los datos de `fruit_features.csv`, entrena un modelo de **Random Forest** y realiza una validación cruzada para medir su precisión. Al finalizar, guarda el modelo entrenado en `fruit_classifier.pkl` y genera gráficas de importancia de características y matriz de confusión.

### 3. 🔮 Predicción e Interfaz
*   **`predict_image.py`**: Permite probar el modelo cargando una imagen individual. Utiliza GrabCut para la segmentación y muestra la predicción junto con las probabilidades de cada clase.
*   **`bot_telegram.py`**: Implementa un **Bot de Telegram** que permite enviar fotos de frutas y recibir la clasificación instantáneamente. Es ideal para pruebas en tiempo real desde dispositivos móviles.

### 4. 🔬 Herramientas Avanzadas y Pruebas
*   **`python visualizar_caracteristicas_sam2.py`**: Una herramienta de diagnóstico avanzada que utiliza **SAM2 (Segment Anything Model 2)** de Meta y **Descriptores de Fourier** para analizar la geometría de las frutas con alta precisión. Visualiza contornos reconstruidos y canales de color.
*   **`segmentacion.py`**: Contiene las funciones modulares de segmentación (Otsu y GrabCut) que son utilizadas por los otros scripts. Puede ejecutarse por separado para visualizar el resultado de la máscara de una imagen.
*   **`prueba_gemini.py`**: Un script de demostración simplificado que utiliza un Árbol de Decisión con datos de ejemplo para explicar de forma didáctica cómo funciona la clasificación basada en forma y color.

---

## 🚀 Flujo de Trabajo Recomendado

1.  **Preparación**: Coloca las imágenes de entrenamiento en la carpeta `dataset/`, organizadas en subcarpetas por clase (ej: `dataset/apple/`).
2.  **Extracción**: Ejecuta `fruit_feature_extractor.py` para generar el CSV con los datos numéricos.
3.  **Entrenamiento**: Ejecuta `train_model.py` para generar el archivo `.pkl` del modelo.
4.  **Uso**: Utiliza `predict_image.py` para predicciones rápidas o inicia `bot_telegram.py` para una interfaz interactiva.

---

## 🛠️ Requisitos Principales
*   Python 3.x
*   OpenCV (`cv2`)
*   NumPy & Pandas
*   Scikit-Learn
*   Scikit-Image
*   Joblib
*   Python-telegram-bot (para el bot)
*   PyTorch & SAM2 (solo para el script de visualización avanzada)

---
*Desarrollado como parte del proyecto de Máquina Separadora de Frutas.*
