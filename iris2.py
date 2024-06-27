from sklearn.model_selection import train_test_split

# Separar as features e o target
X = df.drop('class', axis=1)
y = df['class']

# Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mostrar as formas dos conjuntos de dados
print(f'Treinamento: {X_train.shape}, Teste: {X_test.shape}')
