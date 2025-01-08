import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression

# Cargar los datos
df_2024 = pd.read_csv("C:/Users/user/Desktop/Proyect ML - UN/Proyecto SISMO - Codigo/BBDD Servicio Geologico/2024.csv", delimiter=";")
df_2018 = pd.read_csv("C:/Users/user/Desktop/Proyect ML - UN/Proyecto SISMO - Codigo/BBDD Servicio Geologico/2018.csv", delimiter=";")

df2 = df_2024
df1 = df_2018

df2 = df2.drop(columns=["QuakeML","Fases","Mapa","Estado","Tipo Mag.","Fases.1"])
df1 = df1.drop(columns=["ESTADO","MAGNITUD Mw","# FASES","DEPARTAMENTO"])

df1 = df1.rename(columns={"FECHA":"Fecha",
                            "HORA_UTC":"Hora",
                            "LATITUD (grados)":"Latitud",
                            "LONGITUD (grados)":"Longitud",
                            "PROFUNDIDAD (Km)":"Profundidad (Km)",
                            "MAGNITUD Ml":"Magnitud (MI)",
                            "DEPARTAMENTO":"Departamento",
                            "MUNICIPIO":"Municipio",
                            "RMS (Seg)":"Rms (Seg) / Velocidad del sismo",
                            "GAP (grados)":"Gap (Grados)",
                            "ERROR LATITUD (Km)":"Error Latitud (Km)",
                            "ERROR LONGITUD (Km)":"Error Longitud (Km)",
                            "ERROR PROFUNDIDAD (Km)":"Error Profundidad (Km)",
                            "INTENSIDAD":"Intensidad"})

df2 = df2.rename(columns={ "Fecha-Hora":"Fecha",
                            "(UTC)":"Hora",
                            "Lat(°)":"Latitud",
                            "Long(°)":"Longitud",
                            "Prof(Km)":"Profundidad (Km)",
                            "Mag.":"Magnitud (MI)",
                            "Rms(Seg)":"Rms (Seg) / Velocidad del sismo",
                            "Gap(°)":"Gap (Grados)",
                            "Error Lat(Km)":"Error Latitud (Km)",
                            "Error Long(Km)":"Error Longitud (Km)",
                            "Error Prof(Km)":"Error Profundidad (Km)",
                            "Region":"Municipio",
                            "Intensidad":"Intensidad"  
    
}
)   

# %%
df1 = df1[["Fecha","Hora","Latitud","Longitud","Profundidad (Km)","Magnitud (MI)","Rms (Seg) / Velocidad del sismo","Gap (Grados)","Error Latitud (Km)","Error Longitud (Km)","Error Profundidad (Km)","Municipio","Intensidad"]]

# %%
df2 = df2[["Fecha","Hora","Latitud","Longitud","Profundidad (Km)","Magnitud (MI)","Rms (Seg) / Velocidad del sismo","Gap (Grados)","Error Latitud (Km)","Error Longitud (Km)","Error Profundidad (Km)","Municipio","Intensidad"]]

# %%
# Para colocar los datos en minuscula y delimitar la columna muncipio y quitar la palabra Antioquia
df2["Municipio"] = df2["Municipio"].str.split(" - ").str[0]
df1["Municipio"] = df1["Municipio"].str.lower()
df1["Municipio"].str.lower().inplace=True

df1["Gap (Grados)"] = df1["Gap (Grados)"].fillna(0.0).astype("int64")

df = pd.concat([df1,df2], axis=0)

df["Rms (Seg) / Velocidad del sismo"].fillna(df["Rms (Seg) / Velocidad del sismo"].median(), inplace=True)
df["Error Latitud (Km)"].fillna(df["Error Latitud (Km)"].median(), inplace=True)
df["Error Longitud (Km)"].fillna(df["Error Longitud (Km)"].median(), inplace=True)
df["Error Profundidad (Km)"].fillna(df["Error Profundidad (Km)"].median(), inplace=True)

# %%
df["Fecha"] = pd.to_datetime(df["Fecha"], format="%d/%m/%Y")

# %%
df["Hora"] = pd.to_datetime(df["Hora"], format="%H:%M:%S").dt.time

# División de datos
X = df.drop(columns="Intensidad")
y = df["Intensidad"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocesamiento
categorical_features = ["Fecha", "Hora", "Municipio"]
numerical_features = [
    "Latitud", "Longitud", "Profundidad (Km)", "Magnitud (MI)",
    "Rms (Seg) / Velocidad del sismo", "Gap (Grados)",
    "Error Latitud (Km)", "Error Longitud (Km)", "Error Profundidad (Km)"
]

numerical_transformer = MinMaxScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("encoder", categorical_transformer, categorical_features),
        ("numerica", numerical_transformer, numerical_features)
    ]
)

# Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("feature_selection", SelectKBest(score_func=f_classif, k=10)),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

# Optimización de hiperparámetros
param_grid = {
    "feature_selection__k": [5, 7, 10],
    "classifier__C": [0.1, 1, 10],
    "classifier__solver": ["liblinear", "lbfgs"]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=2, scoring="balanced_accuracy", n_jobs=-1, refit=True)
grid_search.fit(X_train, y_train)

# Guardar el modelo entrenado en un archivo .pkl
with open("BBDD Servicio Geologico/sismo_predictor.pkl", "wb") as file:
    pickle.dump(grid_search, file)


