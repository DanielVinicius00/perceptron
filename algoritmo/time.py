import time
import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        y_ = np.where(y <= 0, 0, 1)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self._activation_function(linear_output)

                update = self.learning_rate * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self._activation_function(linear_output)
        return y_predicted

    def _activation_function(self, x):
        return np.where(x >= 0, 1, 0)

# Exemplo de utilização do Perceptron com medição de tempo
if __name__ == "__main__":
    X_train = ...  # Defina seus dados de treinamento aqui
    y_train = ...  # Defina seus rótulos de treinamento aqui

    start_time = time.time()
    perceptron = Perceptron(learning_rate=0.01, n_iters=1000)
    perceptron.fit(X_train, y_train)
    end_time = time.time()

    processing_time = end_time - start_time
    print(f'Tempo de processamento: {processing_time:.2f} segundos')

    X_test = ...  # Defina seus dados de teste aqui
    y_test = ...  # Defina seus rótulos de teste aqui

    predictions = perceptron.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Acurácia: {accuracy:.2f}')
