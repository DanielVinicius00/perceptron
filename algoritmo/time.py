import time

# Instanciar e treinar o Perceptron com medição de tempo
start_time = time.time()
perceptron = Perceptron(learning_rate=0.01, n_iters=1000)
perceptron.fit(X_train, y_train)
end_time = time.time()

# Calcular o tempo de processamento
processing_time = end_time - start_time
print(f'Tempo de processamento: {processing_time:.2f} segundos')

# Fazer previsões e calcular a acurácia
predictions = perceptron.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Acurácia: {accuracy:.2f}')
