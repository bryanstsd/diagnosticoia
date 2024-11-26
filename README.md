# Sistema de Diagnóstico Médico con Machine Learning

Sistema de apoyo al diagnóstico médico basado en múltiples modelos de machine learning, con énfasis en Random Forest como modelo principal.

## 📝 Descripción
Sistema de apoyo al diagnóstico médico que utiliza técnicas de machine learning para analizar síntomas y signos vitales, proporcionando predicciones sobre posibles condiciones médicas. El sistema implementa múltiples modelos de clasificación, con Random Forest como modelo principal, y proporciona una interfaz web interactiva para su uso.

## ✨ Características
- Diagnóstico basado en múltiples síntomas y signos vitales
- Comparación de 5 modelos diferentes de ML
- Interfaz web intuitiva con Streamlit
- Visualizaciones interactivas de resultados
- Análisis de importancia de características
- Evaluación de modelos con métricas detalladas
- Sistema de histórico de diagnósticos

## 🛠️ Tecnologías Utilizadas
- Python 3.9+
- Scikit-learn para modelos ML
- Streamlit para interfaz web
- Pandas y NumPy para procesamiento de datos
- Plotly y Seaborn para visualizaciones
- SQLite para almacenamiento de históricos

## 📋 Requisitos
- Python 3.9 o superior
- pip (gestor de paquetes de Python)
- Git (opcional, para clonar el repositorio)
- 4GB RAM mínimo recomendado
- Espacio en disco: 500MB mínimo

## 🚀 Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/usuario/sistema-diagnostico-medico.git
cd sistema-diagnostico-medico
```

2. Crear y activar entorno virtual (opcional pero recomendado):
```bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Unix o MacOS:
source venv/bin/activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## 💻 Uso

1. Entrenar los modelos:
```bash
python entrenar_modelo.py
```

2. Iniciar la aplicación:
```bash
streamlit run app.py
```

3. Acceder a la interfaz web:
- Abrir navegador en `http://localhost:8501`


## 🤖 Modelo Implementado
Random Forest (Principal)
  - Mejor balance precisión/interpretabilidad
  - Manejo natural de características mixtas
  - Importancia de características



## 📊 Métricas y Evaluación
- Accuracy
- Precision
- Recall
- F1-Score
- ROC Curves
- Matrices de Confusión
- Importancia de Características


## 🙋‍♂️ Autores
- Jorge Iván Cujia Luquez,  Bryan Jose Salas Altahona, Jose Enrique Alvarez Lara



## 🔍 Estado del Proyecto
- Versión actual: 1.0.0
- Última actualización: Noviembre 2024
- Estado: En desarrollo activo
