# ANALISIS CANCER DE MAMA

#LIBRERIAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sns.set_style('whitegrid')

#CARGAR DATASET

cancer = load_breast_cancer()

df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target

print(df.head())
print(df.shape)

#ESTADISTICAS

print(df.describe())
print(df['target'].value_counts())

#VISUALIZACION

plt.figure(figsize=(10,7))
sns.scatterplot(
    x='mean radius',
    y='mean texture',
    hue='target',
    data=df
)
plt.show()

#SEPARA ENTRENAMIENTO Y PRUEBA

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

#MODELO KNN

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("Accuray KNN:", accuracy_knn)

#MATRZI DE CONFUSION KNN

cm_knn = confusion_matrix(y_test, y_pred_knn)
print(cm_knn)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues')
plt.show()

#ARBOL DE DECISION

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Accuracy Arbol:", accuracy_dt)

#MATRIZ ARBOL

cm_dt = confusion_matrix(y_test, y_pred_dt)
print(cm_dt)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens')
plt.show()

#OVERFITTING

dt_overfit = DecisionTreeClassifier(random_state=42)
dt_overfit.fit(X_train, y_train)

train_acc =accuracy_score(
    y_train,
    dt_overfit.predict(X_train)
)
test_acc = accuracy_score(
    y_test,
    dt_overfit.predict(X_test)
)
print("Train:", train_acc)
print("Test:", test_acc)

