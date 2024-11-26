# modules/model.py

import joblib
import streamlit as st

@st.cache_resource
def load_model(model_path='models/modelo_random_forest.pkl'):
    """
    Carga el modelo de Random Forest desde el archivo especificado.
    """
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"El archivo '{model_path}' no se encontró.")
        st.stop()
