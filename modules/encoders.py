# modules/encoders.py

import joblib
import streamlit as st

@st.cache_resource
def load_label_encoders(encoders_path='models/label_encoders.pkl'):
    """
    Carga los LabelEncoders desde el archivo especificado.
    """
    try:
        encoders = joblib.load(encoders_path)
        return encoders
    except FileNotFoundError:
        st.error(f"El archivo '{encoders_path}' no se encontró.")
        st.stop()
