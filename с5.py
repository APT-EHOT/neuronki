import numpy as np
import pandas as pd
from numpy.linalg import norm


class NeuralNetwork():
    def __init__(self, eta, n_iter, cl_am):
        self.eta = eta
        self.n_iter = n_iter
        self.cl_am = cl_am

    def calculate_distance(self, row):
        eu_dists = list()
        for i in range(len(self.w)):
            eu_dists.append(norm(row - self.w))
        min_elem_idx = eu_dists.index(min(eu_dists))

        return min_elem_idx

    def set_clusters_value(self, amount, dim):
        clusters = list()
        for i in range(amount):
            clusters.append(np.random.uniform(0, 1, dim))

        return clusters

    def fit(self, x):
        self.w = self.set_clusters_value(self.cl_am, x.shape[1])
        for _ in range(self.n_iter):
            for row in x:
                min_elem_idx = self.calculate_distance(row)
                self.w[min_elem_idx] = self.w[min_elem_idx] + self.eta * (row - self.w[min_elem_idx])
            self.eta /= 2

        return self

    def predict(self, row):
        return self.calculate_distance(row)


df = np.loadtxt('data5.txt', delimiter=',')

nn = NeuralNetwork(0.1, 50, 2).fit(df)

idx = 1
print(nn.predict(df[idx]))

row = np.array([[0.75, 0.75], [0.25, 0.25]])
idx = 0
print(nn.predict(row[idx]))

print(nn.w)