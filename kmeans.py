import math
import random

c1 = 0.0
c2 = 0.0


def kmeans(X):
    global c1, c2
    c1, c2 = generate_centroids()
    for i in range(10):
        print(c1, c2)
        c1d, c2d = calculate_distance(X)
        calculate_centroids(c1d, c2d)
        mean = [sum(c1d) / len(c1d), sum(c2d) / len(c2d)]
        sd = [calc_sd(mean[0], c1d), calc_sd(mean[1], c2d)]
        print(sd)
    return


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



kmeans([random.uniform(0, 1) for _ in range(100)])
