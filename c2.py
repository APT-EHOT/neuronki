import numpy as np
import pandas as pd

df = pd.read_csv('data.txt')

X = df.drop('Y', axis=1).to_numpy()

Y = df['Y'].to_numpy()

print(df)


class NeuralNetwork():
    def __init__(self, eta, n_iter, e=0):
        self.eta = eta
        self.n_iter = n_iter
        self.e = e

    def output(self, x):
        return x @ self.w + self.intercept

    def fit(self, x, y):
        self.w = np.zeros(x.shape[1])
        self.intercept = 0
        for _ in range(self.n_iter):
            for row, result in zip(x, y):
                e_tmp = None
                while e_tmp == None or abs(e_tmp) > 0:
                    e_tmp = result - self.predict(row)
                    delta = self.eta * e_tmp
                    self.w += delta * row
                    self.intercept += delta
        return self

    def predict(self, x):
        return np.where(self.output(x) > 0, 1, 0)


nn = NeuralNetwork(1, 10, 0).fit(X, Y)
row = 1
print(nn.predict(X[row]), X[row])

print(nn.output(X[row]))

print(nn.w, nn.intercept)