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

    k = int(sys.argv[1])
    r = int(sys.argv[2])

    ms = list(range(r))
    S = list(range(r))
    sse = np.zeros((r))
    for i in range(r):
        ms[i], S[i], sse[i] = kmeans(data, k)

    lsse = np.argmin(sse)
    print(f'''The minimum SSE for the {r} runs is: {sse[lsse]}''')
    plot(S[lsse], ms[lsse], 'mplot.png')

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


def twonorm(x, m):
    sum = 0
    for i in range(x.size):
        sum += pow(x[i] - m[i], 2)

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