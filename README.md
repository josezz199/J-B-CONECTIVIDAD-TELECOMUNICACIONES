📊 Predicción de Cobertura Móvil en Colombia (MinTIC)
Este proyecto utiliza técnicas de Machine Learning para predecir el nivel de cobertura móvil (Baja/Media/Alta) en diferentes municipios y departamentos de Colombia, basado en datos oficiales de MinTIC.

🚀 Guía de Ejecución Rápida
Para que el proyecto funcione correctamente, siga estos tres pasos:

1. Instalación de Dependencias
Asegúrese de tener un entorno virtual activo y ejecute:

Bash
pip install -r requirements.txt
2. Entrenamiento del Modelo (Pipeline)
Este script procesa los datos de la carpeta data/, aplica el preprocesamiento modular de src/ y genera el archivo del modelo entrenado:

Bash
EJECUTAR : python src/train.py
Al finalizar, verá un archivo llamado model.pkl en la carpeta models/.

3. Lanzamiento de la Aplicación (Interfaz)
Para interactuar con el modelo a través de una interfaz web, ejecute:

Bash
EJECUTAR streamlit run app/app.py
📂 Estructura del Proyecto
El repositorio está organizado de forma modular siguiendo estándares de la industria:

app/: Contiene la aplicación interactiva desarrollada en Streamlit.

data/: Almacena el dataset original de MinTIC en formato CSV.

models/: Contiene el modelo serializado (.pkl) listo para producción.

notebooks/: Análisis Exploratorio de Datos (EDA) y experimentos con Plotly.

src/: Core del proyecto. Contiene el ColumnTransformer y la configuración del pipeline para asegurar que el preprocesamiento sea idéntico en entrenamiento y en la app.

🛠️ Tecnologías Utilizadas
Python 3.x

Scikit-Learn: Uso de Pipelines y ColumnTransformer para evitar el Data Leakage.

XGBoost / RandomForest: Algoritmos de clasificación para la predicción.

Streamlit: Despliegue de la interfaz de usuario.

Joblib: Serialización del modelo.

👨‍💻 Autor
Jose Ballesteros
