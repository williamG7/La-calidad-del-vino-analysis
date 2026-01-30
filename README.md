# La Calidad del Vino - Análisis y Clasificación

![Wine Quality](https://vadevi.elmon.cat/app/uploads/sites/18/2022/02/pinot_noir_cat-800x405.webp)

## Descripción

Proyecto de análisis y clasificación de la calidad del vino utilizando **Machine Learning** con árboles de decisión. Este trabajo implementa un modelo de clasificación basado en características químicas de vinos para predecir su nivel de calidad (baja, media o alta).

## Objetivo

A partir de un conjunto de datos con atributos químicos de diversos vinos, crear un **árbol de decisión** que permita:
- Identificar qué grado de impacto tiene cada variable independiente en la clasificación
- Predecir la calidad del vino basándose en sus propiedades químicas
- Visualizar y entender el proceso de toma de decisiones del modelo

## Dataset

El proyecto utiliza un dataset sintético diseñado para tareas de clasificación de calidad del vino, que incluye **1,000 muestras** con las siguientes características:

### Variables Independientes (Features)
- **`fixed_acidity`**: Nivel de acidez fija
- **`residual_sugar`**: Nivel de azúcar residual después de la fermentación
- **`alcohol`**: Contenido de alcohol (%)
- **`density`**: Densidad del líquido

### Variable Dependiente (Target)
- **`quality_label`**: Clase de calidad del vino
  - `low` (baja)
  - `medium` (media)
  - `high` (alta)

🔗 **Fuente del dataset**: [wine_quality_classification.csv](https://raw.githubusercontent.com/raimonizard/datasets/refs/heads/main/wine_quality_classification.csv)

## Metodología

### 1. **Análisis Exploratorio de Datos (EDA)**
- Exploración de la estructura del dataset
- Identificación de tipos de datos y valores nulos
- Análisis de correlaciones entre variables
- Visualización con matriz de correlación (heatmap)

### 2. **Construcción del Modelo**
- Implementación de `DecisionTreeClassifier` de Scikit-Learn
- Configuración con profundidad máxima de 4 niveles
- Entrenamiento con el dataset completo

### 3. **Visualización del Árbol**
- Representación gráfica de la estructura del árbol de decisión
- Visualización de los nodos y las reglas de decisión

### 4. **Interpretación de Nodos**
Análisis de métricas en cada nodo:
- **Gini**: Índice de impureza del nodo
- **Samples**: Número de muestras que alcanzan ese nodo
- **Value**: Distribución de clases en cada etapa

### 5. **Validación y Predicciones**
- Pruebas de predicción con registros del dataset original
- Comparación de valores reales vs predichos
- Cálculo de métricas de rendimiento

### 6. **Inferencia sobre Nuevos Casos**
- Simulación de nuevos vinos con atributos personalizados
- Predicción de calidad para casos sintéticos

## Tecnologías y Librerías

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

### Dependencias principales:
- **pandas**: Manipulación y análisis de datos
- **matplotlib**: Visualización de gráficos
- **seaborn**: Visualizaciones estadísticas avanzadas
- **scikit-learn**: Algoritmos de Machine Learning

## Cómo Usar

### Opción 1: Google Colab (Recomendado)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/williamG7/La-calidad-del-vino-analysis/blob/main/La_calidad_del_vino_GuzmanWilliam.ipynb)

### Opción 2: Instalación Local

1. **Clonar el repositorio**
```bash
git clone https://github.com/williamG7/La-calidad-del-vino-analysis.git
cd La-calidad-del-vino-analysis
```

2. **Instalar dependencias**
```bash
pip install pandas matplotlib seaborn scikit-learn
```

3. **Ejecutar el notebook**
```bash
jupyter notebook La_calidad_del_vino_GuzmanWilliam.ipynb
```

## Resultados Principales

El modelo de árbol de decisión permite:
- ✅ Clasificar vinos en tres categorías de calidad
- ✅ Identificar las variables más relevantes para la clasificación
- ✅ Visualizar el proceso de decisión de forma interpretable
- ✅ Realizar predicciones sobre nuevos casos

## Estructura del Proyecto

```
La-calidad-del-vino-analysis/
│
├── La_calidad_del_vino_GuzmanWilliam.ipynb   # Notebook principal
└── README.md                                  # Este archivo
```

## Autor

**William Guzmán**

## Licencia

Este proyecto está disponible como código abierto para fines educativos y de aprendizaje.

---

⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub
