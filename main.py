import os
import importlib.util
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron as SKPerceptron
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# Função para carregar um algoritmo de um arquivo Python
def carregar_algoritmo(nome_arquivo):
    caminho = os.path.join('algoritmo', nome_arquivo)
    spec = importlib.util.spec_from_file_location(nome_arquivo, caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo

# Carregar algoritmos disponíveis na pasta 'algoritmo'
algoritmos = []
for arquivo in os.listdir('algoritmo'):
    if arquivo.endswith('.py') and arquivo != '__init__.py':
        algoritmo = carregar_algoritmo(arquivo)
        algoritmos.append(algoritmo)

# Carregar dataset IRIS
iris = load_iris()
df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df_iris['class'] = iris.target
X_iris = df_iris.drop('class', axis=1)
y_iris = df_iris['class']

# Dividir dados IRIS em treinamento e teste
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

# Normalizar dados IRIS
scaler_iris = StandardScaler()
X_train_iris = scaler_iris.fit_transform(X_train_iris)
X_test_iris = scaler_iris.transform(X_test_iris)

# Carregar dataset Wine
wine = load_wine()
df_wine = pd.DataFrame(data=wine.data, columns=wine.feature_names)
df_wine['class'] = wine.target
X_wine = df_wine.drop('class', axis=1)
y_wine = df_wine['class']

# Dividir dados Wine em treinamento e teste
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(X_wine, y_wine, test_size=0.2, random_state=42)

# Normalizar dados Wine
scaler_wine = StandardScaler()
X_train_wine = scaler_wine.fit_transform(X_train_wine)
X_test_wine = scaler_wine.transform(X_test_wine)

# Comparar com Perceptron do Scikit-Learn para IRIS
sk_perceptron_iris = SKPerceptron(eta0=0.01, max_iter=1000, random_state=42)
sk_perceptron_iris.fit(X_train_iris, y_train_iris)
sk_predictions_iris = sk_perceptron_iris.predict(X_test_iris)
sk_accuracy_iris = accuracy_score(y_test_iris, sk_predictions_iris)
print(f'Acurácia do Perceptron do Scikit-Learn para IRIS: {sk_accuracy_iris:.2f}')

# Comparar com Perceptron do Scikit-Learn para Wine
sk_perceptron_wine = SKPerceptron(eta0=0.01, max_iter=1000, random_state=42)
sk_perceptron_wine.fit(X_train_wine, y_train_wine)
sk_predictions_wine = sk_perceptron_wine.predict(X_test_wine)
sk_accuracy_wine = accuracy_score(y_test_wine, sk_predictions_wine)
print(f'Acurácia do Perceptron do Scikit-Learn para Wine: {sk_accuracy_wine:.2f}')

# Validar modelos com Cross-Validation
def validar_modelo(modelo, X, y, cv=5):
    scores = cross_val_score(modelo, X, y, cv=cv)
    return scores

# Validar Perceptron com IRIS usando Cross-Validation
scores_perceptron_iris = validar_modelo(sk_perceptron_iris, X_iris, y_iris)
print(f'Acurácias do Perceptron do Scikit-Learn com Cross-Validation para IRIS: {scores_perceptron_iris}')
print(f'Acurácia Média do Perceptron do Scikit-Learn com Cross-Validation para IRIS: {scores_perceptron_iris.mean():.2f}')

# Validar Perceptron com Wine usando Cross-Validation
scores_perceptron_wine = validar_modelo(sk_perceptron_wine, X_wine, y_wine)
print(f'Acurácias do Perceptron do Scikit-Learn com Cross-Validation para Wine: {scores_perceptron_wine}')
print(f'Acurácia Média do Perceptron do Scikit-Learn com Cross-Validation para Wine: {scores_perceptron_wine.mean():.2f}')

# Executar algoritmos personalizados para IRIS
for algoritmo in algoritmos:
    perceptron_personalizado = algoritmo.Perceptron(learning_rate=0.01, n_iters=1000)
    perceptron_personalizado.fit(X_train_iris, y_train_iris)
    predictions_personalizado = perceptron_personalizado.predict(X_test_iris)
    accuracy_personalizado = accuracy_score(y_test_iris, predictions_personalizado)
    print(f'Acurácia do Perceptron Personalizado para IRIS: {accuracy_personalizado:.2f}')

# Executar algoritmos personalizados para Wine
for algoritmo in algoritmos:
    perceptron_personalizado = algoritmo.Perceptron(learning_rate=0.01, n_iters=1000)
    perceptron_personalizado.fit(X_train_wine, y_train_wine)
    predictions_personalizado = perceptron_personalizado.predict(X_test_wine)
    accuracy_personalizado = accuracy_score(y_test_wine, predictions_personalizado)
    print(f'Acurácia do Perceptron Personalizado para Wine: {accuracy_personalizado:.2f}')
