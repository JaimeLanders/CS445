import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import sys

MINDELTA = 0.0001

def main():
    print("Welcome to Homework 3")
    infile = 'cluster_dataset.txt'

    data = np.loadtxt(infile)

#    fig, ax = plt.subplots()
#    plot(data, 'dplot.png')

    mode = int(sys.argv[1])

    if mode == 1:
        k = int(sys.argv[2])
        r = int(sys.argv[3])

        ms = list(range(r))
        S = list(range(r))
        sse = np.zeros((r))
        for i in range(r):
            ms[i], S[i], sse[i] = kmeans(data, k)

        lsse = np.argmin(sse)
        print(f'''The minimum SSE for the {r} runs is: {sse[lsse]}''')
        plot(S[lsse], ms[lsse], 'mplot.png')
    elif mode == 2:
        cmeans(data)
    else:
        print('Usage: python hw3 mode{1 = k-Means, 2 = FCM} K/c r')
        return -1

    return 0


def assignment(x, m, k, S):
    for p in range(x.shape[0]):
        t = np.zeros((k))
        for i in range(k):
            t[i] = twonorm(x[p], m[i])
        s = np.argmin(t)
        xs = (x[p, 0], x[p, 1])
        S[s].append(xs)

    return S


def cmeans(x):
    m = 2
    x = np.array([[1, 2], [0, -1]])
    w = np.array([[0.4, 0.6], [0.7, 0.3]])

    # Choose a number of clusters: c (a hyperparameter).
#    c = sys.argv[2]

    # Initially assign coefficients randomly to each data point for being in the
    # clusters (these are the initial membership grades).

    # Repeat until the algorithm has converged/stopping condition:
    while True:
        # (I) Compute the centroid for each cluster (m-step).
        c = cmstep(x, w, m)

        # (II) For each data point, compute its coefficients/membership grades
        # for being in the clusters (e-step).
        w = cestep(x, w, m, c)

        break # temp

    print()
    return 0


def cestep(x, w, m, c):
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            sum = 0
            for k in range(c.shape[0]):
                sum += pow(twonorm(x[i], c[j], 1) /
                               (twonorm(x[i], c[k], 1)), (2 / (m - 1)))

            w[i, j] = 1 / sum
    return w


def cmstep(x, w, m):
    c = np.zeros((2, 2))
    for k in range(c.shape[0]):
        numsum = 0
        densum = 0
        for i in range(x.shape[0]):
            numsum += w[i, k] ** m * x[i]
            densum += w[i, k] ** m
            print()
        c[k] = numsum/densum

    return c


def kmeans(x, k):
    # 1. Select K points as initial centroids
    idx = np.random.choice(x.shape[0], size=k)
    m = x[idx, :]

    # 2. repeat until Centroid do not change:
    ml = np.zeros((k))
    it = 0
    S = []
    while True:
        # Form K clusters by assigning each point to its closest centroid
        S = []
        for i in range(k):
            t = [(m[i, 0], m[i, 1])]
            S.append(t)
        S = assignment(x, m, k, S)

        # Recompute the centroid of each cluster
        m = update(m, S)

        if np.any(ml != 0.0):
            t = m - ml
            if max(t.max(), t.min(), key=abs) < 0.001:
                break
        ml = copy.deepcopy(m)

    sse = sumofsquares(S)

    return m, S, sse


def plot(S, m, fname):
    fig, ax = plt.subplots()
    for i in range(len(S)):
        for j in range(len(S[i])):
            Six, Siy = zip(*S[i])
        plt.scatter(Six, Siy)
    plt.scatter(m[:, 0], m[:, 1], marker='+')
    plt.savefig(fname)


def sumofsquares(S):
    sse = 0
    sumx = 0
    sumy = 0
    for i in range(len(S)):
        for j in range(1, len(S[i])):
            sumx += pow(S[i][j][0] - S[i][0][0], 2)
            sumy += pow(S[i][j][1] - S[i][0][1], 2)

    sse = sumx + sumy
    return sse


def twonorm(x, m, d):
    sum = 0
    for i in range(x.size):
        sum += pow(x[i] - m[i], 2)
    if d == 1:
        return math.sqrt(sum)
    else:
        return sum


def update(m, S):
    for i in range(m.shape[0]):
        sum = [0, 0]
        for j in range(1, len(S[i])):
            sum[0] += S[i][j][0]
            sum[1] += S[i][j][1]
        t =  (sum[0] / len(S[i]), sum[1] / len(S[i]))
        m[i] = t
    return m


if __name__ == '__main__':
    main()