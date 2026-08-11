# ==============================
# ANÁLISIS DEL DATASET CÁNCER DE MAMA
# ==============================

# 1. Importar librerías
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 2. Cargar el dataset
data = load_breast_cancer()

# Convertir los datos a DataFrame
X = pd.DataFrame(data.data, columns=data.feature_names)

# Variable objetivo
y = pd.Series(data.target, name="target")


# 3. Información general del dataset
print("Número de observaciones y variables:")
print(X.shape)

print("\nNombre de las clases:")
print(data.target_names)

print("\nPrimeras filas del dataset:")
print(X.head())

print("\nDistribución de clases:")
print(y.value_counts())


# 4. Estadísticas descriptivas
print("\nEstadísticas descriptivas:")
print(X.describe())


# 5. Agregar la clase al DataFrame para análisis
df = X.copy()
df["target"] = y
df["class"] = df["target"].map({
    0: "Maligno",
    1: "Benigno"
})

print("\nPromedio de variables por clase:")
print(df.groupby("class").mean())


# 6. Gráfica de dispersión
plt.figure(figsize=(7,5))

for clase, nombre in [(0, "Maligno"), (1, "Benigno")]:
    datos = df[df["target"] == clase]
    plt.scatter(
        datos["mean radius"],
        datos["mean texture"],
        label=nombre,
        alpha=0.7
    )

plt.xlabel("Mean radius")
plt.ylabel("Mean texture")
plt.title("Relación entre mean radius y mean texture")
plt.legend()
plt.show()


# 7. Histograma
plt.figure(figsize=(7,5))

for clase, nombre in [(0, "Maligno"), (1, "Benigno")]:
    datos = df[df["target"] == clase]
    plt.hist(
        datos["worst perimeter"],
        bins=18,
        alpha=0.6,
        label=nombre
    )

plt.xlabel("Worst perimeter")
plt.ylabel("Frecuencia")
plt.title("Distribución de worst perimeter por clase")
plt.legend()
plt.show()


# 8. División de datos: entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTamaño del conjunto de entrenamiento:")
print(X_train.shape)

print("\nTamaño del conjunto de prueba:")
print(X_test.shape)


# 9. Definir modelos
modelos = {
    "Regresión logística": Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", LogisticRegression(max_iter=1000))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", KNeighborsClassifier(n_neighbors=5))
    ]),

    "SVM lineal": Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", SVC(kernel="linear"))
    ]),

    "Árbol de decisión": DecisionTreeClassifier(
        random_state=42,
        max_depth=4
    )
}


# 10. Entrenar y evaluar modelos
resultados = []

for nombre, modelo in modelos.items():
    print("\n==============================")
    print("Modelo:", nombre)
    print("==============================")

    # Entrenar
    modelo.fit(X_train, y_train)

    # Predecir
    y_pred = modelo.predict(X_test)

    # Calcular accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))

    resultados.append({
        "Modelo": nombre,
        "Accuracy": accuracy
    })


# 11. Comparar modelos
resultados_df = pd.DataFrame(resultados)

print("\nComparación final de modelos:")
print(resultados_df.sort_values(by="Accuracy", ascending=False))


# 12. Gráfica de comparación de modelos
plt.figure(figsize=(8,5))
plt.bar(resultados_df["Modelo"], resultados_df["Accuracy"])
plt.xlabel("Modelo")
plt.ylabel("Accuracy")
plt.title("Comparación de accuracy entre modelos")
plt.xticks(rotation=30)
plt.ylim(0.8, 1.0)
plt.show()
