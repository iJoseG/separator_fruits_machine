import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==========================================================
# CONFIGURACIÓN
# ==========================================================

CSV_PATH = "fruit_features.csv"
MODEL_OUTPUT = "fruit_classifier.pkl"

# ==========================================================
# CARGAR DATASET
# ==========================================================

df = pd.read_csv(CSV_PATH)

print("\nDataset cargado correctamente.")
print("Clases encontradas:", df["label"].unique())

# ==========================================================
# SEPARAR FEATURES Y LABELS
# ==========================================================

X = df.drop("label", axis=1)
y = df["label"]

# ==========================================================
# DIVISIÓN TRAIN / TEST
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nDatos divididos correctamente.")
print("Entrenamiento:", len(X_train))
print("Prueba:", len(X_test))

# ==========================================================
# CREAR MODELO
# ==========================================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42
)

# ==========================================================
# VALIDACIÓN CRUZADA
# ==========================================================

cv_scores = cross_val_score(model, X, y, cv=5)

print("\nValidación cruzada (5-fold):")
print("Accuracy promedio:", cv_scores.mean())
print("Desviación estándar:", cv_scores.std())

# ==========================================================
# ENTRENAR
# ==========================================================

model.fit(X_train, y_train)

# ==========================================================
# EVALUACIÓN
# ==========================================================

y_pred = model.predict(X_test)

print("\nAccuracy en test:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================================================
# MATRIZ DE CONFUSIÓN
# ==========================================================

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.tight_layout()
plt.show()

# ==========================================================
# IMPORTANCIA DE FEATURES
# ==========================================================

importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

plt.figure(figsize=(8,6))
importances[:10].plot(kind='bar')
plt.title("Top 10 Features Más Importantes")
plt.tight_layout()
plt.show()

# ==========================================================
# GUARDAR MODELO
# ==========================================================

joblib.dump(model, MODEL_OUTPUT)

print(f"\nModelo guardado correctamente como: {MODEL_OUTPUT}")
