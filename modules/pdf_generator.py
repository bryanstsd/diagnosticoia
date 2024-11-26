# modules/pdf_generator.py

from fpdf import FPDF
import streamlit as st

def generar_pdf(datos_paciente, enfermedad_predicha):
    """
    Genera un informe PDF con los datos del paciente y la predicción.
    """
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(0, 10, 'Datos del Paciente:', ln=True)
        for clave, valor in datos_paciente.items():
            pdf.cell(0, 10, f'- {clave}: {valor}', ln=True)
        pdf.cell(0, 10, '', ln=True)
        pdf.cell(0, 10, 'Predicción de la Enfermedad:', ln=True)
        pdf.cell(0, 10, f'{enfermedad_predicha}', ln=True)
        pdf.cell(0, 10, '', ln=True)
        pdf.cell(0, 10, '***', ln=True)
        pdf.multi_cell(0, 10, 'Aviso Importante: Este informe es informativo y no reemplaza una consulta médica profesional. Se recomienda consultar a un profesional de la salud.')

        # Obtener el contenido del PDF como bytes
        pdf_output = pdf.output(dest='S').encode('latin1')  # 'latin1' es compatible con FPDF
        return pdf_output
    except Exception as e:
        st.error(f"Error al generar el PDF: {e}")
        return None
