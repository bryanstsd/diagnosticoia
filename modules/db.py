# modules/db.py

import sqlite3
import streamlit as st

def initialize_db(db_path='historial.db'):
    """
    Crea la tabla 'historial' en la base de datos SQLite si no existe.
    """
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute('''
        CREATE TABLE IF NOT EXISTS historial (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TEXT,
            edad INTEGER,
            genero TEXT,
            presion_arterial TEXT,
            colesterol TEXT,
            fiebre TEXT,
            tos TEXT,
            fatiga TEXT,
            dificultad_respirar TEXT,
            enfermedad_predicha TEXT
        )
        ''')
        conn.commit()

def guardar_diagnostico(fecha, edad, genero, presion_arterial, colesterol, fiebre, tos, fatiga, dificultad_respirar, enfermedad_predicha, db_path='historial.db'):
    """
    Inserta un nuevo diagnóstico en la base de datos.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            c = conn.cursor()
            c.execute('''
                INSERT INTO historial (fecha, edad, genero, presion_arterial, colesterol, fiebre, tos, fatiga, dificultad_respirar, enfermedad_predicha)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                fecha,
                edad,
                genero,
                presion_arterial,
                colesterol,
                fiebre,
                tos,
                fatiga,
                dificultad_respirar,
                enfermedad_predicha
            ))
            conn.commit()
    except Exception as e:
        st.error(f"Error al guardar en la base de datos: {e}")
