import numpy as np
import pandas as pd


class NeuralNetwork():
    def __init__(self, eta, n_iter):
        self.eta = eta
        self.n_iter = n_iter

    def output(self, x):
        return x @ self.w + self.intercept

    def fit(self, x, y):
        self.w = np.zeros(x.shape[1])
        self.intercept = 0
        for _ in range(self.n_iter):
            for row, result in zip(x, y):
                if self.predict(row) != result:
                    for j in range(len(self.w)):
                        self.w[j] += self.eta * row[j] * (result - self.predict(row))
                    self.intercept += self.eta * (result - self.predict(row))
        return self

    def predict(self, x):
        return np.where(self.output(x) > 0, 1, 0)


df = pd.read_csv('data.txt')
X = df.drop('Y', axis=1).to_numpy()
Y = df['Y'].to_numpy()

nn = NeuralNetwork(1, 10).fit(X, Y)
row = 3
print(f"X = {X[row]}, Y = {nn.predict(X[row])}")
print(f"Веса = {nn.w}, Т = {nn.intercept}")