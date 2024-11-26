import streamlit as st

# Debe ser la primera línea de Streamlit
st.set_page_config(
    page_title="Sistema de Diagnóstico Médico",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import datetime
import os
import sqlite3
import json
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from PIL import Image
import base64
import io
import time
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


# Asegurar que existan los directorios necesarios
def crear_directorios():
    """Crea los directorios necesarios si no existen"""
    directorios = ['assets', 'backups']
    for directorio in directorios:
        if not os.path.exists(directorio):
            os.makedirs(directorio)

# Crear los directorios al inicio
crear_directorios()

# Funciones de utilidad
def load_css():
    """Carga los estilos CSS desde el archivo"""
    with open('assets/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def initialize_session_state():
    """Inicializa variables en session_state"""
    if 'current_patient' not in st.session_state:
        st.session_state.current_patient = None
    if 'diagnosis_history' not in st.session_state:
        st.session_state.diagnosis_history = []
    if 'model_metrics' not in st.session_state:
        st.session_state.model_metrics = None


def load_model_metrics():
    """Carga las métricas del modelo desde el archivo JSON"""
    try:
        with open('metrics/model_metrics.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def crear_conexion_db():
    """Crea una conexión a la base de datos SQLite"""
    return sqlite3.connect('historial.db')


def guardar_diagnostico_db(datos_paciente, diagnostico, probabilidad):
    """Guarda el diagnóstico en la base de datos"""
    try:
        conn = crear_conexion_db()
        cursor = conn.cursor()

        # Crear tabla si no existe
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS historial (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TIMESTAMP,
            edad INTEGER,
            genero TEXT,
            presion_arterial TEXT,
            colesterol TEXT,
            fiebre BOOLEAN,
            tos BOOLEAN,
            fatiga BOOLEAN,
            dificultad_respiratoria BOOLEAN,
            diagnostico TEXT,
            probabilidad FLOAT,
            nivel_riesgo TEXT
        )
        ''')

        # Calcular nivel de riesgo
        sintomas = sum([
            datos_paciente['fever'],
            datos_paciente['cough'],
            datos_paciente['fatigue'],
            datos_paciente['difficulty_breathing']
        ])
        valores_anormales = sum([
            datos_paciente['blood_pressure'] != 'Normal',
            datos_paciente['cholesterol_level'] != 'Normal'
        ])
        riesgo_total = sintomas + valores_anormales

        if riesgo_total <= 2:
            nivel_riesgo = "Bajo"
        elif riesgo_total <= 4:
            nivel_riesgo = "Moderado"
        else:
            nivel_riesgo = "Alto"

        # Insertar datos
        cursor.execute('''
        INSERT INTO historial (
            fecha, edad, genero, presion_arterial, colesterol,
            fiebre, tos, fatiga, dificultad_respiratoria,
            diagnostico, probabilidad, nivel_riesgo
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.datetime.now(),
            datos_paciente['age'],
            datos_paciente['gender'],
            datos_paciente['blood_pressure'],
            datos_paciente['cholesterol_level'],
            datos_paciente['fever'],
            datos_paciente['cough'],
            datos_paciente['fatigue'],
            datos_paciente['difficulty_breathing'],
            diagnostico,
            probabilidad,
            nivel_riesgo
        ))

        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error al guardar en la base de datos: {str(e)}")
        return False
    finally:
        conn.close()


def generar_pdf(datos_informe):
    """Genera un PDF con el informe del diagnóstico"""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

        # Crear buffer para el PDF
        buffer = io.BytesIO()

        # Crear el documento
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []

        # Estilos
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30
        )

        # Título
        elements.append(Paragraph("Informe de Diagnóstico Médico", title_style))
        elements.append(Spacer(1, 12))

        # Datos del paciente
        patient_data = [
            ["Datos del Paciente", ""],
            ["Edad:", f"{datos_informe['paciente']['age']} años"],
            ["Género:", datos_informe['paciente']['gender']],
            ["Presión Arterial:", datos_informe['paciente']['blood_pressure']],
            ["Colesterol:", datos_informe['paciente']['cholesterol_level']]
        ]

        # Crear tabla con los datos
        t = Table(patient_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        elements.append(t)
        elements.append(Spacer(1, 20))

        # Diagnóstico
        elements.append(Paragraph("Diagnóstico", title_style))
        elements.append(Paragraph(f"Diagnóstico Principal: {datos_informe['diagnostico']}", styles['Normal']))
        elements.append(Paragraph(f"Probabilidad: {datos_informe['probabilidad']:.1%}", styles['Normal']))
        elements.append(Paragraph(f"Nivel de Riesgo: {datos_informe['nivel_riesgo']}", styles['Normal']))

        # Generar PDF
        doc.build(elements)

        # Obtener el valor del buffer
        pdf = buffer.getvalue()
        buffer.close()

        return pdf
    except Exception as e:
        st.error(f"Error al generar el PDF: {str(e)}")
        return None


def procesar_entrada(datos_paciente, label_encoders):
    """Procesa los datos de entrada para el modelo"""
    try:
        # Definir el orden exacto de las características como en el entrenamiento
        feature_order = [
            'Age',
            'Gender',
            'Blood Pressure',
            'Cholesterol Level',
            'Fever',
            'Cough',
            'Fatigue',
            'Difficulty Breathing'
        ]

        # Crear el diccionario de datos manteniendo el orden
        datos = {
            'Age': [float(datos_paciente['age'])],  # Convertir edad a float
            'Gender': [datos_paciente['gender']],
            'Blood Pressure': [datos_paciente['blood_pressure']],
            'Cholesterol Level': [datos_paciente['cholesterol_level']],
            'Fever': ['Yes' if datos_paciente['fever'] else 'No'],
            'Cough': ['Yes' if datos_paciente['cough'] else 'No'],
            'Fatigue': ['Yes' if datos_paciente['fatigue'] else 'No'],
            'Difficulty Breathing': ['Yes' if datos_paciente['difficulty_breathing'] else 'No']
        }

        # Crear DataFrame asegurando el orden correcto de las columnas
        entrada = pd.DataFrame(datos)[feature_order]

        # Aplicar label encoding solo a las columnas categóricas
        categorical_columns = [
            'Gender',
            'Blood Pressure',
            'Cholesterol Level',
            'Fever',
            'Cough',
            'Fatigue',
            'Difficulty Breathing'
        ]

        for columna in categorical_columns:
            if columna in label_encoders:
                try:
                    entrada[columna] = label_encoders[columna].transform(entrada[columna])
                except Exception as e:
                    st.error(f"Error al codificar la columna {columna}: {str(e)}")
                    st.write(f"Valores únicos en {columna}:", entrada[columna].unique())
                    st.write(f"Clases conocidas por el encoder:", label_encoders[columna].classes_)
                    raise

        # Verificar que todas las columnas se procesaron correctamente
        st.write("Debug - DataFrame final:", entrada)
        return entrada

    except Exception as e:
        st.error(f"Error al procesar los datos: {str(e)}")
        st.write("Datos del paciente:", datos_paciente)
        st.write("Label encoders disponibles:", list(label_encoders.keys()))
        st.write("Tipos de datos:", entrada.dtypes)
        return None


def cargar_modelo():
    """Carga el modelo entrenado y sus etiquetas"""
    try:
        import joblib
        # Cargar el modelo
        model = joblib.load('models/modelo_random_forest.pkl')
        label_encoders = joblib.load('models/label_encoders.pkl')

        # Debug: Mostrar información del modelo
        st.write("Características esperadas por el modelo:", model.feature_names_in_)
        st.write("Label encoders disponibles:", list(label_encoders.keys()))

        return model, label_encoders
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None, None


def realizar_prediccion(datos_paciente):
    """Realiza la predicción usando el modelo cargado"""
    # Cargar modelo
    model, label_encoders = cargar_modelo()
    if model is None or label_encoders is None:
        return "No disponible", 0.0, {}

    try:
        # Procesar datos de entrada
        entrada = procesar_entrada(datos_paciente, label_encoders)
        if entrada is None:
            return "Error en procesamiento", 0.0, {}

        # Debug: Verificar que las características coincidan
        st.write("Características del modelo:", model.feature_names_in_)
        st.write("Características de entrada:", entrada.columns.tolist())

        # Realizar predicción
        prediccion = model.predict(entrada)[0]
        probabilidades = model.predict_proba(entrada)[0]

        # Obtener la etiqueta de la predicción y su probabilidad
        enfermedad = label_encoders['Disease'].inverse_transform([prediccion])[0]
        probabilidad = max(probabilidades)

        # Obtener top 3 predicciones
        indices_top = probabilidades.argsort()[-3:][::-1]
        predicciones_top = {
            label_encoders['Disease'].inverse_transform([i])[0]: probabilidades[i]
            for i in indices_top
        }

        return enfermedad, probabilidad, predicciones_top

    except Exception as e:
        st.error(f"Error al realizar la predicción: {str(e)}")
        import traceback
        st.write("Traceback completo:", traceback.format_exc())
        return "Error en predicción", 0.0, {}

def mostrar_diagnostico():
    st.markdown('<h1 class="main-header">🩺 Diagnóstico Médico</h1>', unsafe_allow_html=True)

    # Contenedor principal
    with st.container():
        col1, col2 = st.columns([2, 1])

        with col1:
            # Datos del paciente
            st.markdown("### 📋 Información del Paciente")

            data_col1, data_col2 = st.columns(2)

            with data_col1:
                age = st.number_input('Edad', min_value=1, max_value=120, value=30)
                gender = st.radio('Género', ['Male', 'Female'])

            with data_col2:
                blood_pressure = st.selectbox(
                    'Presión Arterial',
                    ['Low', 'Normal', 'High'],
                    help="Seleccione el nivel de presión arterial"
                )
                cholesterol_level = st.selectbox(
                    'Nivel de Colesterol',
                    ['Low', 'Normal', 'High'],
                    help="Seleccione el nivel de colesterol"
                )

            # Síntomas
            st.markdown("### 🤒 Síntomas")
            symptoms_cols = st.columns(4)

            with symptoms_cols[0]:
                fever = st.checkbox('Fiebre 🌡️')
            with symptoms_cols[1]:
                cough = st.checkbox('Tos 😷')
            with symptoms_cols[2]:
                fatiga = st.checkbox('Fatiga 😫')
            with symptoms_cols[3]:
                difficulty_breathing = st.checkbox('Dificultad Respiratoria 🫁')

    # Botón de diagnóstico
    if st.button('Realizar Diagnóstico 🔍', use_container_width=True):
        with st.spinner('Analizando datos...'):
            # Preparar datos del paciente
            datos_paciente = {
                'age': age,
                'gender': gender,
                'blood_pressure': blood_pressure,
                'cholesterol_level': cholesterol_level,
                'fever': fever,
                'cough': cough,
                'fatigue': fatiga,
                'difficulty_breathing': difficulty_breathing
            }

            # Realizar predicción
            diagnostico, probabilidad, predicciones_top = realizar_prediccion(datos_paciente)

            # Mostrar resultados
            st.markdown("### 📊 Resultados del Diagnóstico")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                    <div class="diagnosis-card">
                        <h4>Diagnóstico Principal</h4>
                        <div class="diagnosis-result">{}</div>
                        <div class="diagnosis-probability">Probabilidad: {:.1%}</div>
                    </div>
                """.format(diagnostico, probabilidad), unsafe_allow_html=True)

                # Mostrar predicciones alternativas
                if predicciones_top:
                    st.markdown("#### Diagnósticos Alternativos:")
                    for enfermedad, prob in predicciones_top.items():
                        if enfermedad != diagnostico:
                            st.markdown(f"- {enfermedad}: {prob:.1%}")

            with col2:
                nivel_riesgo = calcular_nivel_riesgo(datos_paciente)
                st.markdown(f"""
                    <div class="risk-card">
                        <h4>Nivel de Riesgo</h4>
                        <div class="risk-level">{nivel_riesgo}</div>
                    </div>
                """, unsafe_allow_html=True)

            # Guardar diagnóstico
            if guardar_diagnostico_db(datos_paciente, diagnostico, probabilidad):
                st.success("Diagnóstico guardado correctamente")

            # Generar PDF
            pdf_bytes = generar_pdf({
                'paciente': datos_paciente,
                'diagnostico': diagnostico,
                'probabilidad': probabilidad,
                'nivel_riesgo': nivel_riesgo
            })

            if pdf_bytes:
                st.download_button(
                    label="📄 Descargar Informe",
                    data=pdf_bytes,
                    file_name="informe_medico.pdf",
                    mime="application/pdf"
                )


def calcular_nivel_riesgo(datos_paciente):
    """Calcula el nivel de riesgo basado en los datos del paciente"""
    # Contar síntomas presentes
    sintomas = sum([
        datos_paciente['fever'],
        datos_paciente['cough'],
        datos_paciente['fatigue'],
        datos_paciente['difficulty_breathing']
    ])

    # Contar valores anormales
    valores_anormales = sum([
        datos_paciente['blood_pressure'] != 'Normal',
        datos_paciente['cholesterol_level'] != 'Normal'
    ])

    # Calcular riesgo total
    riesgo_total = sintomas + valores_anormales

    if riesgo_total <= 2:
        return "Bajo"
    elif riesgo_total <= 4:
        return "Moderado"
    else:
        return "Alto"


def mostrar_analisis():
    st.markdown('<h1 class="main-header">📊 Análisis de Datos</h1>', unsafe_allow_html=True)

    # Cargar datos históricos
    try:
        conn = crear_conexion_db()
        query = """
        SELECT 
            fecha,
            edad,
            genero,
            presion_arterial,
            colesterol,
            fiebre,
            tos,
            fatiga,
            difficulty_breathing,
            diagnostico,
            probabilidad,
            nivel_riesgo
        FROM historial
        ORDER BY fecha DESC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        if len(df) == 0:
            st.info("No hay datos históricos disponibles para analizar. Realiza algunos diagnósticos primero.")
            return

        # Convertir fecha a datetime
        df['fecha'] = pd.to_datetime(df['fecha'])

        # 1. Métricas Generales
        st.markdown("### 📈 Métricas Generales")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Diagnósticos",
                len(df),
                f"{len(df[df['fecha'] > pd.Timestamp.now() - pd.Timedelta(days=7)])} nuevos esta semana"
            )

        with col2:
            casos_altos = len(df[df['nivel_riesgo'] == 'Alto'])
            st.metric(
                "Casos Alto Riesgo",
                casos_altos,
                f"{(casos_altos / len(df) * 100):.1f}% del total"
            )

        with col3:
            precision_media = df['probabilidad'].mean()
            st.metric(
                "Confianza Media",
                f"{precision_media:.1%}"
            )

        with col4:
            edad_media = df['edad'].mean()
            st.metric(
                "Edad Media",
                f"{edad_media:.1f} años"
            )

        # 2. Distribución de Diagnósticos
        st.markdown("### 🎯 Distribución de Diagnósticos")

        # Gráfico de diagnósticos más comunes
        diagnosticos = df['diagnostico'].value_counts()
        fig_diagnosticos = go.Figure(data=[
            go.Bar(
                x=diagnosticos.index,
                y=diagnosticos.values,
                marker_color='#1E88E5'
            )
        ])
        fig_diagnosticos.update_layout(
            title="Diagnósticos más Comunes",
            xaxis_title="Diagnóstico",
            yaxis_title="Número de Casos",
            height=400
        )
        st.plotly_chart(fig_diagnosticos, use_container_width=True)

        # 3. Análisis Temporal
        st.markdown("### 📅 Análisis Temporal")

        # Gráfico de tendencias temporales
        df_temporal = df.resample('D', on='fecha')['diagnostico'].count().reset_index()
        fig_temporal = go.Figure(data=[
            go.Scatter(
                x=df_temporal['fecha'],
                y=df_temporal['diagnostico'],
                mode='lines+markers',
                name='Casos por día',
                line=dict(color='#1E88E5', width=2),
                marker=dict(size=6)
            )
        ])
        fig_temporal.update_layout(
            title="Tendencia de Diagnósticos en el Tiempo",
            xaxis_title="Fecha",
            yaxis_title="Número de Casos",
            height=400
        )
        st.plotly_chart(fig_temporal, use_container_width=True)

        # 4. Distribución por Nivel de Riesgo
        st.markdown("### ⚠️ Distribución por Nivel de Riesgo")

        # Gráfico de distribución de riesgos
        riesgos = df['nivel_riesgo'].value_counts()
        colores = {'Alto': '#dc3545', 'Moderado': '#ffc107', 'Bajo': '#28a745'}

        fig_riesgos = go.Figure(data=[
            go.Pie(
                labels=riesgos.index,
                values=riesgos.values,
                marker=dict(colors=[colores.get(x, '#1E88E5') for x in riesgos.index]),
                hole=.3
            )
        ])
        fig_riesgos.update_layout(
            title="Distribución de Niveles de Riesgo",
            height=400
        )
        st.plotly_chart(fig_riesgos, use_container_width=True)

        # 5. Tabla de Resumen
        st.markdown("### 📋 Resumen Detallado")

        # Preparar datos para la tabla
        resumen = pd.DataFrame({
            'Métrica': [
                'Total de Diagnósticos',
                'Casos de Alto Riesgo',
                'Diagnóstico más Común',
                'Edad Promedio',
                'Confianza Promedio',
                'Casos esta Semana'
            ],
            'Valor': [
                str(len(df)),
                f"{casos_altos} ({(casos_altos / len(df) * 100):.1f}%)",
                f"{diagnosticos.index[0]} ({diagnosticos.values[0]} casos)",
                f"{edad_media:.1f} años",
                f"{precision_media:.1%}",
                str(len(df[df['fecha'] > pd.Timestamp.now() - pd.Timedelta(days=7)]))
            ]
        })

        st.table(resumen)

        # 6. Opciones de Exportación
        st.markdown("### 💾 Exportar Datos")
        col1, col2 = st.columns(2)

        with col1:
            # Exportar a CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar CSV",
                data=csv,
                file_name="analisis_diagnosticos.csv",
                mime="text/csv",
            )

        with col2:
            if st.button("📊 Generar Reporte PDF"):
                st.info("Funcionalidad de reporte PDF en desarrollo")

    except Exception as e:
        st.error(f"Error al cargar o analizar los datos: {str(e)}")
        st.write("Error detallado:", str(e))
        import traceback
        st.write("Traceback:", traceback.format_exc())


def crear_grafica_distribucion(df):
    """Crea gráfica de distribución de diagnósticos"""
    diagnosticos = df['diagnostico'].value_counts()

    fig = go.Figure(data=[
        go.Bar(
            x=diagnosticos.index,
            y=diagnosticos.values,
            marker_color='#1E88E5',
            hovertemplate='<b>%{x}</b><br>' +
                          'Casos: %{y}<extra></extra>'
        )
    ])

    fig.update_layout(
        title='Distribución de Diagnósticos',
        xaxis_title='Diagnóstico',
        yaxis_title='Número de Casos',
        template='plotly_white',
        height=400
    )

    return fig


def crear_grafica_tendencias(df):
    """Crea gráfica de tendencias temporales"""
    df['fecha'] = pd.to_datetime(df['fecha'])
    tendencia = df.groupby(df['fecha'].dt.date)['diagnostico'].count().reset_index()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=tendencia['fecha'],
        y=tendencia['diagnostico'],
        mode='lines+markers',
        name='Casos por día',
        line=dict(color='#1E88E5', width=2),
        marker=dict(size=6)
    ))

    fig.update_layout(
        title='Tendencia de Diagnósticos',
        xaxis_title='Fecha',
        yaxis_title='Número de Casos',
        template='plotly_white',
        height=400
    )

    return fig


def crear_grafica_riesgo(df):
    """Crea gráfica de distribución de niveles de riesgo"""
    riesgos = df['nivel_riesgo'].value_counts()

    colores = {
        'Bajo': '#4CAF50',
        'Moderado': '#FFC107',
        'Alto': '#F44336'
    }

    fig = go.Figure(data=[
        go.Pie(
            labels=riesgos.index,
            values=riesgos.values,
            marker_colors=[colores.get(r, '#1E88E5') for r in riesgos.index],
            hole=.3
        )
    ])

    fig.update_layout(
        title='Distribución de Niveles de Riesgo',
        template='plotly_white',
        height=400
    )

    return fig


def mostrar_evaluacion():
    st.markdown('<h1 class="main-header">📈 Evaluación del Modelo</h1>', unsafe_allow_html=True)

    # Cargar métricas del modelo
    try:
        with open('metrics/model_metrics.json', 'r') as f:
            metrics = json.load(f)
    except FileNotFoundError:
        st.error("No se encontraron métricas del modelo. Ejecute primero el entrenamiento.")
        return
    except Exception as e:
        st.error(f"Error al cargar las métricas: {str(e)}")
        return

    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Accuracy",
            f"{metrics.get('accuracy', 0):.1%}",
            help="Precisión global del modelo"
        )

    with col2:
        st.metric(
            "Precisión",
            f"{metrics.get('precision_weighted', 0):.1%}",
            help="Precisión ponderada del modelo"
        )

    with col3:
        st.metric(
            "Recall",
            f"{metrics.get('recall_weighted', 0):.1%}",
            help="Exhaustividad ponderada del modelo"
        )

    with col4:
        st.metric(
            "F1 Score",
            f"{metrics.get('f1_weighted', 0):.1%}",
            help="Media armónica entre precisión y recall"
        )

    # Visualizaciones
    st.markdown("### 📊 Visualizaciones")

    # Cargar y mostrar imágenes guardadas
    col1, col2 = st.columns(2)

    with col1:
        try:
            st.image('plots/confusion_matrix.png', caption='Matriz de Confusión')
        except:
            st.error("No se pudo cargar la matriz de confusión")

    with col2:
        try:
            st.image('plots/feature_importance.png', caption='Importancia de Características')
        except:
            st.error("No se pudo cargar el gráfico de importancia de características")

    try:
        st.image('plots/roc_curves.png', caption='Curvas ROC')
    except:
        st.error("No se pudo cargar el gráfico de curvas ROC")

    # Mostrar detalles adicionales
    if 'classification_report' in metrics:
        st.markdown("### 📋 Reporte de Clasificación Detallado")
        df_report = pd.DataFrame(metrics['classification_report']).drop('support', errors='ignore')
        st.dataframe(df_report.style.background_gradient(cmap='RdYlGn', axis=None))


def crear_matriz_confusion(metrics):
    """Crea la visualización de la matriz de confusión"""
    if 'confusion_matrix' not in metrics or 'class_names' not in metrics:
        return go.Figure()

    matrix = metrics['confusion_matrix']
    labels = metrics['class_names']

    fig = ff.create_annotated_heatmap(
        z=matrix,
        x=labels,
        y=labels,
        colorscale='Blues',
        showscale=True
    )

    fig.update_layout(
        title='Matriz de Confusión',
        xaxis_title="Predicción",
        yaxis_title="Valor Real",
        height=500
    )

    return fig


def crear_curvas_roc(metrics):
    """Crea la visualización de las curvas ROC"""
    if 'roc_curves' not in metrics:
        return go.Figure()

    fig = go.Figure()

    for class_name, curve in metrics['roc_curves'].items():
        fig.add_trace(go.Scatter(
            x=curve['fpr'],
            y=curve['tpr'],
            name=f'{class_name} (AUC = {curve["auc"]:.2f})',
            mode='lines'
        ))

    # Agregar línea diagonal de referencia
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        name='Referencia',
        mode='lines',
        line=dict(dash='dash', color='gray')
    ))

    fig.update_layout(
        title='Curvas ROC',
        xaxis_title='Tasa de Falsos Positivos',
        yaxis_title='Tasa de Verdaderos Positivos',
        height=500
    )

    return fig


def crear_importancia_caracteristicas(metrics):
    """Crea la visualización de importancia de características"""
    if 'feature_importance' not in metrics:
        return go.Figure()

    importance = metrics['feature_importance']
    features = list(importance.keys())
    values = list(importance.values())

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker_color='#1E88E5'
    ))

    fig.update_layout(
        title='Importancia de Características',
        xaxis_title='Importancia',
        yaxis_title='Característica',
        height=500
    )

    return fig


def mostrar_matriz_confusion_ejemplo():
    """Muestra una matriz de confusión de ejemplo"""
    matrix = np.array([
        [45, 5, 2],
        [3, 35, 4],
        [2, 3, 30]
    ])
    labels = ['Clase A', 'Clase B', 'Clase C']

    fig = ff.create_annotated_heatmap(
        z=matrix,
        x=labels,
        y=labels,
        colorscale='Blues',
        showscale=True
    )

    fig.update_layout(
        title='Matriz de Confusión (Ejemplo)',
        xaxis_title="Predicción",
        yaxis_title="Valor Real",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)
    st.info("Nota: Esta es una matriz de confusión de ejemplo.")


def mostrar_curvas_roc_ejemplo():
    """Muestra curvas ROC de ejemplo"""
    fig = go.Figure()

    # Generar datos de ejemplo
    fpr = np.linspace(0, 1, 100)
    tpr1 = np.array([min(1, x * 1.5) for x in fpr])
    tpr2 = np.array([min(1, x * 1.3) for x in fpr])
    tpr3 = np.array([min(1, x * 1.1) for x in fpr])

    fig.add_trace(go.Scatter(
        x=fpr, y=tpr1,
        name='Clase A (AUC = 0.85)',
        mode='lines'
    ))

    fig.add_trace(go.Scatter(
        x=fpr, y=tpr2,
        name='Clase B (AUC = 0.80)',
        mode='lines'
    ))

    fig.add_trace(go.Scatter(
        x=fpr, y=tpr3,
        name='Clase C (AUC = 0.75)',
        mode='lines'
    ))

    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        name='Referencia',
        mode='lines',
        line=dict(dash='dash', color='gray')
    ))

    fig.update_layout(
        title='Curvas ROC (Ejemplo)',
        xaxis_title='Tasa de Falsos Positivos',
        yaxis_title='Tasa de Verdaderos Positivos',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)
    st.info("Nota: Estas son curvas ROC de ejemplo.")


def mostrar_importancia_caracteristicas_ejemplo():
    """Muestra gráfico de importancia de características de ejemplo"""
    features = [
        'Edad', 'Presión Arterial', 'Colesterol',
        'Fiebre', 'Tos', 'Fatiga',
        'Dificultad Respiratoria', 'Género'
    ]
    importance = [0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04]

    fig = go.Figure(go.Bar(
        x=importance,
        y=features,
        orientation='h',
        marker_color='#1E88E5'
    ))

    fig.update_layout(
        title='Importancia de Características (Ejemplo)',
        xaxis_title='Importancia',
        yaxis_title='Característica',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)
    st.info("Nota: Este es un gráfico de importancia de características de ejemplo.")


def mostrar_metricas_clase_ejemplo():
    """Muestra métricas por clase de ejemplo"""
    data = {
        'Clase': ['A', 'B', 'C'],
        'Precisión': [0.85, 0.82, 0.78],
        'Recall': [0.83, 0.80, 0.75],
        'F1-Score': [0.84, 0.81, 0.76]
    }

    df = pd.DataFrame(data)
    st.dataframe(
        df.style.background_gradient(cmap='RdYlGn', subset=['Precisión', 'Recall', 'F1-Score'])
    )
    st.info("Nota: Estas son métricas de ejemplo.")

def mostrar_historial():
    st.markdown('<h1 class="main-header">📚 Historial de Diagnósticos</h1>', unsafe_allow_html=True)

    # Filtros
    col1, col2, col3 = st.columns(3)

    with col1:
        fecha_inicio = st.date_input(
            "Fecha Inicio",
            value=datetime.datetime.now() - datetime.timedelta(days=30)
        )

    with col2:
        fecha_fin = st.date_input(
            "Fecha Fin",
            value=datetime.datetime.now()
        )

    with col3:
        nivel_riesgo = st.selectbox(
            "Filtrar por Riesgo",
            ["Todos", "Alto", "Moderado", "Bajo"]
        )

    try:
        # Cargar datos
        conn = crear_conexion_db()
        query = """
        SELECT 
            fecha,
            edad,
            genero,
            presion_arterial,
            colesterol,
            diagnostico,
            probabilidad,
            nivel_riesgo
        FROM historial
        WHERE DATE(fecha) BETWEEN ? AND ?
        """

        params = [fecha_inicio.strftime("%Y-%m-%d"), fecha_fin.strftime("%Y-%m-%d")]

        if nivel_riesgo != "Todos":
            query += " AND nivel_riesgo = ?"
            params.append(nivel_riesgo)

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if len(df) == 0:
            st.info("No se encontraron registros para los filtros seleccionados")
            return

        # Mostrar resumen
        st.markdown("### 📊 Resumen del Periodo")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Diagnósticos", len(df))

        with col2:
            st.metric(
                "Edad Promedio",
                f"{df['edad'].mean():.1f} años"
            )

        with col3:
            alto_riesgo = len(df[df['nivel_riesgo'] == 'Alto'])
            st.metric(
                "Casos Alto Riesgo",
                alto_riesgo,
                f"{(alto_riesgo / len(df) * 100):.1f}% del total"
            )

        # Mostrar tabla de registros
        st.markdown("### 📋 Registros")

        # Formatear DataFrame
        df['fecha'] = pd.to_datetime(df['fecha']).dt.strftime('%Y-%m-%d %H:%M')
        df['probabilidad'] = df['probabilidad'].map('{:.1%}'.format)

        # Aplicar colores según nivel de riesgo
        def color_riesgo(val):
            colors = {
                'Alto': 'color: red',
                'Moderado': 'color: orange',
                'Bajo': 'color: green'
            }
            return colors.get(val, '')

        st.dataframe(
            df.style.applymap(color_riesgo, subset=['nivel_riesgo'])
        )

        # Opciones de exportación
        col1, col2 = st.columns(2)

        with col1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Descargar CSV",
                csv,
                "historial_diagnosticos.csv",
                "text/csv",
                key='download-csv'
            )

        with col2:
            if st.button("📊 Generar Reporte"):
                generar_reporte_historial(df)

    except Exception as e:
        st.error(f"Error al cargar el historial: {str(e)}")


def generar_reporte_historial(df):
    """Genera un reporte detallado del historial"""
    st.markdown("### 📑 Reporte Detallado")

    # Estadísticas generales
    st.markdown("#### Estadísticas Generales")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Distribución por Género:**")
        fig = px.pie(df, names='genero')
        st.plotly_chart(fig)

    with col2:
        st.write("**Distribución por Nivel de Riesgo:**")
        fig = px.pie(df, names='nivel_riesgo', color='nivel_riesgo',
                     color_discrete_map={'Alto': 'red', 'Moderado': 'orange', 'Bajo': 'green'})
        st.plotly_chart(fig)

    # Tendencias temporales
    st.markdown("#### Tendencias Temporales")
    df['fecha'] = pd.to_datetime(df['fecha'])
    tendencia = df.groupby(df['fecha'].dt.date)['diagnostico'].count().reset_index()

    fig = px.line(tendencia, x='fecha', y='diagnostico',
                  title='Diagnósticos por Día')
    st.plotly_chart(fig)

    # Estadísticas detalladas
    st.markdown("#### Estadísticas Detalladas")
    st.write(df.describe())


def mostrar_configuracion():
    st.markdown('<h1 class="main-header">⚙️ Configuración</h1>', unsafe_allow_html=True)

    # Configuración del modelo
    st.markdown("### 🤖 Configuración del Modelo")
    with st.expander("Parámetros del Modelo", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            threshold = st.slider(
                "Umbral de Confianza",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                help="Umbral mínimo de confianza para las predicciones"
            )

            st.number_input(
                "Número de Estimadores",
                min_value=50,
                max_value=500,
                value=100,
                step=50,
                help="Número de árboles en el Random Forest"
            )

        with col2:
            st.selectbox(
                "Métrica de Evaluación",
                ["accuracy", "precision", "recall", "f1"],
                help="Métrica principal para evaluar el modelo"
            )

            st.number_input(
                "Profundidad Máxima",
                min_value=3,
                max_value=20,
                value=10,
                help="Profundidad máxima de los árboles"
            )

    # Configuración de la interfaz
    st.markdown("### 🎨 Configuración de la Interfaz")
    with st.expander("Personalización", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            tema = st.selectbox(
                "Tema de la Interfaz",
                ["Claro", "Oscuro", "Sistema"],
                help="Tema visual de la aplicación"
            )

            mostrar_probabilidades = st.checkbox(
                "Mostrar Probabilidades",
                value=True,
                help="Mostrar probabilidades detalladas en los diagnósticos"
            )

        with col2:
            idioma = st.selectbox(
                "Idioma",
                ["Español", "English"],
                help="Idioma de la interfaz"
            )

            mostrar_graficas = st.checkbox(
                "Mostrar Gráficas",
                value=True,
                help="Mostrar visualizaciones en los análisis"
            )

    # Configuración de notificaciones
    st.markdown("### 📧 Configuración de Notificaciones")
    with st.expander("Notificaciones", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            email = st.text_input(
                "Email para Notificaciones",
                help="Dirección de correo para recibir notificaciones"
            )

            notificar_alto_riesgo = st.checkbox(
                "Notificar Casos de Alto Riesgo",
                value=True,
                help="Recibir notificaciones para casos de alto riesgo"
            )

        with col2:
            frecuencia = st.selectbox(
                "Frecuencia de Reportes",
                ["Diario", "Semanal", "Mensual"],
                help="Frecuencia de envío de reportes"
            )

            notificar_actualizaciones = st.checkbox(
                "Notificar Actualizaciones",
                value=True,
                help="Recibir notificaciones sobre actualizaciones del sistema"
            )

    # Configuración de la base de datos
    st.markdown("### 💾 Configuración de la Base de Datos")
    with st.expander("Base de Datos", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.number_input(
                "Retención de Datos (días)",
                min_value=30,
                max_value=365,
                value=180,
                help="Tiempo de retención de datos históricos"
            )

            backup_automatico = st.checkbox(
                "Backup Automático",
                value=True,
                help="Realizar copias de seguridad automáticas"
            )

        with col2:
            frecuencia_backup = st.selectbox(
                "Frecuencia de Backup",
                ["Diario", "Semanal", "Mensual"],
                help="Frecuencia de las copias de seguridad"
            )

            st.text_input(
                "Ubicación de Backup",
                value="./backups",
                help="Directorio para las copias de seguridad"
            )

    # Botones de acción
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("💾 Guardar Configuración", use_container_width=True):
            guardar_configuracion()
            st.success("Configuración guardada correctamente")

    with col2:
        if st.button("🔄 Restaurar Valores por Defecto", use_container_width=True):
            restaurar_configuracion()
            st.success("Configuración restaurada a valores por defecto")

    with col3:
        if st.button("🔄 Actualizar Sistema", use_container_width=True):
            with st.spinner("Actualizando sistema..."):
                # Aquí iría la lógica de actualización
                time.sleep(2)
                st.success("Sistema actualizado correctamente")


def guardar_configuracion():
    """Guarda la configuración actual en un archivo JSON"""
    config = {
        'model': {
            'threshold': st.session_state.get('threshold', 0.5),
            'n_estimators': st.session_state.get('n_estimators', 100),
            'max_depth': st.session_state.get('max_depth', 10),
            'metric': st.session_state.get('metric', 'accuracy')
        },
        'interface': {
            'theme': st.session_state.get('theme', 'Claro'),
            'language': st.session_state.get('language', 'Español'),
            'show_probabilities': st.session_state.get('show_probabilities', True),
            'show_plots': st.session_state.get('show_plots', True)
        },
        'notifications': {
            'email': st.session_state.get('email', ''),
            'notify_high_risk': st.session_state.get('notify_high_risk', True),
            'report_frequency': st.session_state.get('report_frequency', 'Semanal'),
            'notify_updates': st.session_state.get('notify_updates', True)
        },
        'database': {
            'retention_days': st.session_state.get('retention_days', 180),
            'auto_backup': st.session_state.get('auto_backup', True),
            'backup_frequency': st.session_state.get('backup_frequency', 'Semanal'),
            'backup_location': st.session_state.get('backup_location', './backups')
        }
    }

    try:
        with open('config.json', 'w') as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        st.error(f"Error al guardar la configuración: {str(e)}")


def restaurar_configuracion():
    """Restaura la configuración a valores por defecto"""
    config_default = {
        'model': {
            'threshold': 0.5,
            'n_estimators': 100,
            'max_depth': 10,
            'metric': 'accuracy'
        },
        'interface': {
            'theme': 'Claro',
            'language': 'Español',
            'show_probabilities': True,
            'show_plots': True
        },
        'notifications': {
            'email': '',
            'notify_high_risk': True,
            'report_frequency': 'Semanal',
            'notify_updates': True
        },
        'database': {
            'retention_days': 180,
            'auto_backup': True,
            'backup_frequency': 'Semanal',
            'backup_location': './backups'
        }
    }

    try:
        with open('config.json', 'w') as f:
            json.dump(config_default, f, indent=4)

        # Actualizar session_state
        for category, settings in config_default.items():
            for key, value in settings.items():
                st.session_state[key] = value
    except Exception as e:
        st.error(f"Error al restaurar la configuración: {str(e)}")


def crear_conexion_db():
    """Crea una conexión a la base de datos SQLite"""
    try:
        conn = sqlite3.connect('historial.db')
        # Verificar si la tabla existe
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS historial (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fecha TIMESTAMP,
                edad INTEGER,
                genero TEXT,
                presion_arterial TEXT,
                colesterol TEXT,
                fiebre INTEGER,
                tos INTEGER,
                fatiga INTEGER,
                difficulty_breathing INTEGER,
                diagnostico TEXT,
                probabilidad FLOAT,
                nivel_riesgo TEXT
            )
        """)
        conn.commit()
        return conn
    except Exception as e:
        st.error(f"Error al conectar con la base de datos: {str(e)}")
        raise e




def inicializar_base_datos():
    """Inicializa la base de datos si no existe"""
    conn = crear_conexion_db()
    conn.close()

def guardar_diagnostico_db(datos_paciente, diagnostico, probabilidad):
    """Guarda el diagnóstico en la base de datos"""
    try:
        conn = crear_conexion_db()
        cursor = conn.cursor()

        # Calcular nivel de riesgo
        sintomas = sum([
            datos_paciente['fever'],
            datos_paciente['cough'],
            datos_paciente['fatigue'],
            datos_paciente['difficulty_breathing']
        ])
        valores_anormales = sum([
            datos_paciente['blood_pressure'] != 'Normal',
            datos_paciente['cholesterol_level'] != 'Normal'
        ])
        riesgo_total = sintomas + valores_anormales

        if riesgo_total <= 2:
            nivel_riesgo = "Bajo"
        elif riesgo_total <= 4:
            nivel_riesgo = "Moderado"
        else:
            nivel_riesgo = "Alto"

        # Insertar datos
        cursor.execute('''
        INSERT INTO historial (
            fecha,
            edad,
            genero,
            presion_arterial,
            colesterol,
            fiebre,
            tos,
            fatiga,
            difficulty_breathing,
            diagnostico,
            probabilidad,
            nivel_riesgo
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.datetime.now(),
            datos_paciente['age'],
            datos_paciente['gender'],
            datos_paciente['blood_pressure'],
            datos_paciente['cholesterol_level'],
            int(datos_paciente['fever']),
            int(datos_paciente['cough']),
            int(datos_paciente['fatigue']),
            int(datos_paciente['difficulty_breathing']),
            diagnostico,
            probabilidad,
            nivel_riesgo
        ))

        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error al guardar en la base de datos: {str(e)}")
        return False
    finally:
        conn.close()

def main():
    try:
        # Inicializar la base de datos
        inicializar_base_datos()

        # Cargar estilos y configuración inicial
        load_css()
        initialize_session_state()

        # Sidebar navigation
        with st.sidebar:
            st.image('assets/logo.png', width=100)
            st.markdown("### Menú Principal")
            selected = st.radio(
                "",
                ["🏥 Diagnóstico",
                 "📊 Análisis",
                 "📈 Evaluación",
                 "📚 Historial",
                 "⚙️ Configuración"],
                label_visibility="collapsed"
            )

            # Remover el emoji para el procesamiento
            selected = selected.split(" ")[1]

            # Información adicional en el sidebar
            st.markdown("---")
            st.markdown("### Información del Sistema")
            st.info("""
            🏥 Sistema de Diagnóstico Médico v1.0

            ⚕️ Basado en IA

            📊 Actualizaciones diarias
            """)

        # Routing basado en la selección
        if selected == "Diagnóstico":
            mostrar_diagnostico()
        elif selected == "Análisis":
            mostrar_analisis()
        elif selected == "Evaluación":
            mostrar_evaluacion()
        elif selected == "Historial":
            mostrar_historial()
        elif selected == "Configuración":
            mostrar_configuracion()

    except Exception as e:
        st.error(f"Error en la aplicación: {str(e)}")


if __name__ == '__main__':
    main()