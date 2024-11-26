# modules/utils.py

import streamlit as st

def convertir_si_no(valor, columna, label_encoders):
    """
    Convierte una respuesta de 'Yes'/'No' a valores numéricos utilizando el LabelEncoder correspondiente.
    """
    le = label_encoders[columna]
    try:
        return le.transform([valor])[0]
    except ValueError:
        st.error(f"Valor '{valor}' no reconocido en la columna '{columna}'. Se asignará un valor predeterminado (0).")
        return 0  # Valor predeterminado o manejo alternativo

def codificar_variable(valor, columna, label_encoders):
    """
    Codifica una variable categórica utilizando el LabelEncoder correspondiente.
    """
    le = label_encoders[columna]
    try:
        return le.transform([valor])[0]
    except ValueError:
        st.error(f"Valor '{valor}' no reconocido en la columna '{columna}'. Se asignará un valor predeterminado (0).")
        return 0  # Valor predeterminado o manejo alternativo

def cargar_css(file_path='assets/style.css'):
    """
    Carga estilos CSS personalizados.
    """
    try:
        with open(file_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"El archivo de estilos '{file_path}' no se encontró.")
