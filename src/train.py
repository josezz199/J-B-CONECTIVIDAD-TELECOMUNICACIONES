import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os
import sys

from features import clean_data, create_features
from preprocessing import get_pipeline

# Crear carpeta models
os.makedirs("models", exist_ok=True)

# =========================
# 1. CARGA Y LIMPIEZA
# =========================
df = pd.read_csv("data/raw/cobertura.csv")

df = clean_data(df)
df = create_features(df)

# =========================
# 2. VALIDACIÓN DE TARGET
# =========================
print("\nDistribución del target:")
print(df["target"].value_counts())

if df["target"].nunique() < 2:
    print("\n❌ ERROR: Solo hay una clase en el target.")
    print("👉 Ajusta la función de clasificación en features.py")
    sys.exit()

# =========================
# 3. SELECCIÓN DE VARIABLES
# =========================
cols = [
    "PROVEEDOR",
    "DEPARTAMENTO",
    "CABECERA MUNICIPAL",
    "num_tecnologias",
    "target"
]

# Validar columnas
missing_cols = [c for c in cols if c not in df.columns]
if missing_cols:
    print(f"\n❌ Faltan columnas: {missing_cols}")
    sys.exit()

df_model = df[cols].copy()

X = df_model.drop(columns=["target"])
y = df_model["target"]

# =========================
# 4. SPLIT (IMPORTANTE)
# =========================
# stratify evita que desaparezcan clases
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 5. PIPELINE
# =========================
num_cols = ["num_tecnologias"]
cat_cols = ["PROVEEDOR", "DEPARTAMENTO", "CABECERA MUNICIPAL"]

prep = get_pipeline(num_cols, cat_cols)

# =========================
# 6. MODELOS
# =========================
models = {
    "Logistic": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(random_state=42)
}

best_model = None
best_score = 0

for name, model in models.items():

    pipe = Pipeline([
        ("prep", prep),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    print(f"\n🔹 Modelo: {name}")
    print(classification_report(y_test, preds))

    score = pipe.score(X_test, y_test)

    if score > best_score:
        best_score = score
        best_model = pipe

# =========================
# 7. GUARDAR MODELO
# =========================
joblib.dump(best_model, "models/model.pkl")

print("\n✅ Modelo guardado en models/model.pkl")
print(f"🏆 Mejor score: {best_score:.4f}")