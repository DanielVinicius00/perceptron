import pandas as pd
from sklearn.datasets import load_iris

# Carregar o dataset IRIS
iris = load_iris()
# Criar um DataFrame com os dados e adicionar a coluna de classe
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['class'] = iris.target

# Mostrar as primeiras linhas do DataFrame
print(df.head())

from sklearn.model_selection import train_test_split

# Separar as features e o target
X = df.drop('class', axis=1)
y = df['class']

# Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mostrar as formas dos conjuntos de dados
print(f'Treinamento: {X_train.shape}, Teste: {X_test.shape}')

from sklearn.preprocessing import StandardScaler

# Inicializar o normalizador
scaler = StandardScaler()

# Ajustar o normalizador aos dados de treinamento e transformar os dados
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Mostrar os dados normalizados
print(X_train[:5])
