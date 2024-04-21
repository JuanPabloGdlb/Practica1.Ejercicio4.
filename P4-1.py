import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from itertools import combinations
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Cargar los datos
data = pd.read_csv('irisbin.csv', header=None)
X = data.iloc[:, :-3].values
y = data.iloc[:, -3:].values

# Reducción de dimensionalidad con PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Crear y entrenar un perceptrón multicapa
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

# Validar resultados usando leave-k-out
def leave_k_out(X, y, k):
    n = len(X)
    accuracies = []
    for indices in combinations(range(n), k):
        mask = np.ones(n, dtype=bool)
        mask[list(indices)] = False
        X_train, X_val = X[mask], X[~mask]
        y_train, y_val = y[mask], y[~mask]
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_val)
        accuracies.append(accuracy_score(y_val, y_pred))
    return accuracies

# Validar resultados usando leave-one-out
def leave_one_out(X, y):
    return leave_k_out(X, y, 1)

# Calcular el error esperado de clasificación, promedio y desviación estándar
k_out_accuracies = leave_k_out(X, y, 1)  # Usando leave-one-out como ejemplo
error = 1 - np.mean(k_out_accuracies)
avg_accuracy = np.mean(k_out_accuracies)
std_dev = np.std(k_out_accuracies)

print("Error esperado de clasificación:", error)
print("Precisión promedio:", avg_accuracy)
print("Desviación estándar de la precisión:", std_dev)

# Asignar un valor de color único a cada clase
unique_classes = np.unique(y)
class_colors = np.linspace(0, 1, len(unique_classes))

# Crear un diccionario que mapea las clases a los colores
class_to_color = {unique_class: color for unique_class, color in zip(unique_classes, class_colors)}

# Asignar los colores a cada muestra en función de su clase
colors = [class_to_color[class_label[0]] for class_label in y]

# Graficar la distribución de clases para el dataset Irisbin después de la reducción de dimensionalidad
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, cmap='viridis')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Distribución de clases para el dataset Irisbin después de PCA')
plt.show()
