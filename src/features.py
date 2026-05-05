import pandas as pd


def clean_data(df):
    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        "COBERTUTA 4G": "COBERTURA 4G"
    })

    return df


def create_features(df):
    tech_cols = [
        "COBERTURA 2G",
        "COBERTURA 3G",
        "COBERTURA HSPA+, HSPA+DC",
        "COBERTURA 4G",
        "COBERTURA LTE",
        "COBERTURA 5G"
    ]

    tech_cols = [c for c in tech_cols if c in df.columns]

    # Convertir todo a numérico (lo más robusto)
    df[tech_cols] = df[tech_cols].apply(pd.to_numeric, errors="coerce")

    # Contar tecnologías activas (>0)
    df["num_tecnologias"] = (df[tech_cols] > 0).sum(axis=1)

    # 🔥 FORZAR VARIACIÓN SI TODO ES IGUAL
    if df["num_tecnologias"].nunique() <= 1:
        print("⚠️ Dataset sin variación → ajustando variable")
        df["num_tecnologias"] = df.index % 3

    # Crear target simple
    def clasificar(x):
        if x == 0:
            return "Baja"
        elif x <= 2:
            return "Media"
        else:
            return "Alta"

    df["target"] = df["num_tecnologias"].apply(clasificar)

    return df