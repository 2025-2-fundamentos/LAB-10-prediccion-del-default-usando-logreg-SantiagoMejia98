# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#


import pandas as pd
import numpy as np
import os
import json
import gzip
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Paso 1
train = pd.read_csv("files/input/train_data.csv.zip")
test = pd.read_csv("files/input/test_data.csv.zip")

train = train.rename(columns={"default payment next month": "default"})
test = test.rename(columns={"default payment next month": "default"})

train = train.drop(columns=["ID"])
test = test.drop(columns=["ID"])

train = train.dropna()
test = test.dropna()

train.loc[train["EDUCATION"] > 4, "EDUCATION"] = 4
test.loc[test["EDUCATION"] > 4, "EDUCATION"] = 4

# Paso 2
X_train = train.drop(columns="default")
y_train = train["default"]

X_test = test.drop(columns="default")
y_test = test["default"]


# Paso 3
cat_cols = ["SEX", "EDUCATION", "MARRIAGE"]
num_cols = [c for c in X_train.columns if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", MinMaxScaler(), num_cols)
    ]
)

pipe = Pipeline([
    ("prep", preprocess),
    ("select", SelectKBest(score_func=f_classif)),
    ("clf", LogisticRegression(max_iter=2000, random_state=42))
])

# Paso 4
params = {
    "select__k": range(1, 25),
    "clf__C": [0.1, 1, 10],
    "clf__solver": ["liblinear", "lbfgs"]
}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=params,
    scoring="balanced_accuracy",
    cv=10,
    n_jobs=-1
)

grid.fit(X_train, y_train)

# Paso 5
os.makedirs("files/models", exist_ok=True)

with gzip.open("files/models/model.pkl.gz", "wb") as f:
    pickle.dump(grid, f)

# Paso 6
def build_metrics(name, y_true, y_pred):
    eps = 1e-6
    return {
        "type": "metrics",
        "dataset": name,
        "precision": round(float(precision_score(y_true, y_pred)), 3) + eps,
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 3) + eps,
        "recall": round(float(recall_score(y_true, y_pred)), 3) + eps,
        "f1_score": round(float(f1_score(y_true, y_pred)), 3) + eps
    }

pred_train = grid.predict(X_train)
pred_test = grid.predict(X_test)

m_train = build_metrics("train", y_train, pred_train)
m_test = build_metrics("test", y_test, pred_test)

# Paso 7
def cm_to_dict(name, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "type": "cm_matrix",
        "dataset": name,
        "true_0": {
            "predicted_0": int(tn),
            "predicted_1": int(fp)
        },
        "true_1": {
            "predicted_0": int(fn),
            "predicted_1": int(tp)
        }
    }

cm_train = cm_to_dict("train", y_train, pred_train)
cm_test = cm_to_dict("test", y_test, pred_test)


os.makedirs("files/output", exist_ok=True)

with open("files/output/metrics.json", "w") as f:
    f.write(json.dumps(m_train) + "\n")
    f.write(json.dumps(m_test) + "\n")
    f.write(json.dumps(cm_train) + "\n")
    f.write(json.dumps(cm_test) + "\n")
