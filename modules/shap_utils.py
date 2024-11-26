# modules/shap_utils.py

import shap
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st
from modules.model import load_model  # Importa la función para cargar el modelo
from modules.encoders import load_label_encoders

@st.cache_resource
def initialize_explainer():
    """
    Inicializa y devuelve el explainer de SHAP utilizando el modelo cargado.
    """
    try:
        model = load_model()  # Carga el modelo dentro de la función
        explainer = shap.Explainer(model)
        return explainer
    except Exception as e:
        st.error(f"Error al crear el explainer de SHAP: {e}")
        st.stop()

explainer = initialize_explainer()

def generar_shap_values(entrada_df, clases, enfermedad_predicha, X_columns):
    """
    Genera y devuelve los valores SHAP para la clase predicha.
    """
    try:
        shap_values = explainer(entrada_df)
    except Exception as e:
        st.error(f"Error al generar valores SHAP: {e}")
        return None, None

    # Obtener el índice de la clase predicha
    try:
        class_index = np.where(clases == enfermedad_predicha)[0][0]
    except IndexError:
        st.error(f'Error: La clase predicha "{enfermedad_predicha}" no se encuentra en las clases del modelo.')
        return None, None

    # Extraer los valores SHAP para la clase predicha
    if isinstance(shap_values.values, list):
        if class_index < len(shap_values.values):
            shap_values_class = shap_values.values[class_index][0]
            base_value = shap_values.base_values[class_index]
        else:
            st.error(f'Error: class_index {class_index} fuera de rango para shap_values con tamaño {len(shap_values.values)}.')
            shap_values_class = np.zeros(len(X_columns))
            base_value = 0
    else:
        # Para clasificación binaria o regresión
        shap_values_class = shap_values.values[0]
        base_value = shap_values.base_values

    # Asegurarse de que 'shap_values_class' es 1D
    shap_values_class = np.array(shap_values_class).flatten()

    # Asegurarse de que 'entrada' es 1D
    entrada_flat = np.array(entrada_df.iloc[0]).flatten()

    # Crear el DataFrame de importancia
    try:
        df_importance = pd.DataFrame({
            'Característica': X_columns,
            'Valor': entrada_flat,
            'Importancia': shap_values_class
        })
    except ValueError as ve:
        st.error(f"Error al crear el DataFrame de importancia: {ve}")
        return None, None

    # Ordenar por importancia
    df_importance = df_importance.sort_values(by='Importancia', ascending=True)

    return df_importance, base_value
