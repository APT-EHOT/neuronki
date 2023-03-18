import math
import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

c1 = 0.0
c2 = 0.0
def kmeans(X):
    global c1, c2
    c1, c2 = generate_centroids()
    mean = 0.0
    sd = 0.0
    for i in range(10):
        c1d, c2d = calculate_distance(X)
        calculate_centroids(c1d, c2d)
        mean = [sum(c1d) / len(c1d), sum(c2d) / len(c2d)]
        sd = [calc_sd(mean[0], c1d), calc_sd(mean[1], c2d)]
    return mean, sd


def generate_centroids():
    global c1, c2
    c1 = random.uniform(0, 1)
    c2 = random.uniform(0, 1)
    return c1, c2


def calculate_distance(X):
    cluster1dots = []
    cluster2dots = []
    for i in range(len(X)):
        if euclid_dist(X[i], c1) > euclid_dist(X[i], c2):
            cluster2dots.append(X[i])
        else:
            cluster1dots.append(X[i])
    return cluster1dots, cluster2dots


def calculate_centroids(c1dots, c2dots):
    global c1, c2
    c1 = sum(c1dots) / len(c1dots)
    c2 = sum(c2dots) / len(c2dots)


def euclid_dist(x, y):
    return math.sqrt(abs(x*x - y*y))


def calc_sd(mean, dots):
    sd = 0.0
    for i in range(len(dots)):
        sd += (dots[i] - mean) ** 2
    return math.sqrt(sd / len(dots))



class NeuralNetwork:
    def __init__(self, k=2, eta=0.01, n_iter=100):
        self.k = k
        self.eta = eta
        self.n_iter = n_iter

        self.w = np.random.randn(k)
        self.intercept = np.random.randn(1)

    def rbf(self, x, c, s):
        return np.exp(-1 / (2 * s ** 2) * (x - c) ** 2)

    def output(self, x):
        return x @ self.w + self.intercept

    def fit(self, x, y):
        print(x)
        self.centers, self.stds = kmeans(x)

        for n_iter in range(self.n_iter):
            for i in range(x.shape[0]):
                a = np.array([self.rbf(x[i], c, s) for c, s, in zip(self.centers, self.stds)])
                F = self.output(a.T)

                error = -(y[i] - F).flatten()

                self.w = self.w - self.eta * a * error
                self.b = self.intercept - self.eta * error

    def predict(self, x):
        y_pred = []
        for i in range(x.shape[0]):
            a = np.array([self.rbf(x[i], c, s) for c, s, in zip(self.centers, self.stds )])
            F = self.output(a.T)

            y_pred.append(F)
        return np.array(y_pred)

np.random.seed(777)

X = np.random.uniform(0, 1, 100)
X = np.sort(X, axis=0)
NOISE = np.random.uniform(-0.25, 0.25, 100)
Y = np.sin(2 * np.pi * X) + NOISE
nn = NeuralNetwork(eta=0.01, k=2)
nn.fit(X, Y)
y_pred = nn.predict(X)
plt.plot(X, Y, '-o', label='true', color='mediumslateblue')
plt.plot(X, y_pred, '-o', label='predicted', color='orangered')
plt.legend()
plt.tight_layout()
plt.show()