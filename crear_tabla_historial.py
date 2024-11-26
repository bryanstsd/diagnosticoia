import sqlite3

# Conectar a la base de datos (se creará si no existe)
conn = sqlite3.connect('historial.db')
c = conn.cursor()

# Crear la tabla 'historial' si no existe
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
conn.close()

print("Tabla 'historial' creada exitosamente.")
