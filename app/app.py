import streamlit as st
import pandas as pd
import joblib
import os

# 1. CORRECCIÓN DE RUTA: Localiza el modelo sin importar desde dónde lances la app
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

# 2. CARGA DEL MODELO: Con verificación de existencia
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    st.error(f"⚠️ No se encontró el archivo del modelo en: {MODEL_PATH}")
    st.stop()

# 3. INTERFAZ DE USUARIO
st.title("Predicción de Cobertura Móvil")
st.write("Ingrese los datos para predecir el nivel de cobertura")

# Organizar entradas en columnas para que se vea mejor
col1, col2 = st.columns(2)

with col1:
    proveedor = st.selectbox(
        "Proveedor",
        ["CLARO", "MOVISTAR", "TIGO", "WOM", "OTRO"]
    )
    departamento = st.text_input("Departamento", "ANTIOQUIA")

with col2:
    cabecera = st.selectbox(
        "Cabecera Municipal",
        ["SI", "NO"]
    )
    num_tec = st.slider(
        "Número de tecnologías",
        1, 6, 3
    )

# 4. PREPARACIÓN DE DATOS
# Es vital que los nombres de las columnas coincidan exactamente con el entrenamiento
data = pd.DataFrame([{
    "PROVEEDOR": proveedor,
    "DEPARTAMENTO": departamento,
    "CABECERA MUNICIPAL": cabecera,
    "num_tecnologias": num_tec
}])

# 5. PREDICCIÓN
if st.button("Predecir Cobertura"):
    try:
        pred = model.predict(data)
        
        # Mostrar resultado con un diseño llamativo
        st.subheader("Resultado de la Predicción:")
        if pred[0] == "Baja":
            st.warning(f"Nivel de cobertura: {pred[0]}")
        else:
            st.success(f"Nivel de cobertura: {pred[0]}")
            
    except Exception as e:
        st.error(f"Error al procesar los datos: {e}")
        st.info("Asegúrate de que las columnas coincidan con las del entrenamiento.")