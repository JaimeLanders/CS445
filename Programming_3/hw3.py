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

        Ms = list(range(r))
        S = list(range(r))
        sse = np.zeros((r))
        for i in range(r):
            Ms[i], S[i], sse[i] = kmeans(data, k)

        lsse = np.argmin(sse)
        print(f'''The minimum SSE for the {r} runs is: {sse[lsse]}''')
        kplot(S[lsse], Ms[lsse], 'kplot.png')
    elif mode == 2:
        # Choose a number of clusters: c (a hyperparameter).
        c = int(sys.argv[2])
        m = int(sys.argv[3]) # Fuzzifier
        r = int(sys.argv[4])
#        for i in range(r):
        W, C = cmeans(data, c, m)
#        print(f'''The minimum SSE for the {r} runs is: {sse[lsse]}''')
        cplot(data, W, C, 'cplot.png')
    else:
        print('\nUsage: python hw3 mode hyperparam1 (hyperparam2) #runs, (i.e):'
              '       \nk-Means mode (1): python hw3 1 K r'
              '       \nFCM mode (2): python hw3 2 c m r')
        return -1

    return 0


def cestep(X, W, m, C):
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            sum = 0
            for k in range(C.shape[0]):
                sum += pow(twonorm(X[i], C[j], 1) /
                                  (twonorm(X[i], C[k], 1)), (2 / (m - 1)))

            W[i, j] = 1 / sum
    return W


def cmeans(X, c, m):
    # D2L example
#    c = 2
#    m = 2
#    X = np.array([[1, 2], [0, -1]])
#    W = np.array([[0.4, 0.6], [0.7, 0.3]])
#    C = cmstep(X, W, m, c)
#    W = cestep(X, W, m, C)
#    return 0

    # Initially assign coefficients randomly to each data point for being in the
    # clusters (these are the initial membership grades).
#    W = np.random.randint(1, X.shape[0], size = (X.shape[0], c))
    W = np.random.randint(1, 10, size = (X.shape[0], c))
    W = W / np.sum(W)
    C = np.zeros((c, X.shape[1]))

    cl = 0
    # Repeat until the algorithm has converged/stopping condition:
    it = 0
    while True:
        print(it)
        it += 1
        # (I) Compute the centroid for each cluster (m-step).
        C = cmstep(X, W, m, c)

        # (II) For each data point, compute its coefficients/membership grades
        # for being in the clusters (e-step).
        W = cestep(X, W, m, C)

        if np.any(cl != 0.0):
            t = C - cl
            if max(t.max(), t.min(), key=abs) < MINDELTA:
                break
        cl = copy.deepcopy(C)

    return W, C


def cmstep(X, W, m, c):
    C = np.zeros((c, X.shape[1]))
    for k in range(C.shape[0]):
        numsum = 0
        densum = 0
        for i in range(X.shape[0]):
            numsum += W[i, k] ** m * X[i]
            densum += W[i, k] ** m
        C[k] = numsum/densum

    return C


def cplot(X, W, C, fname):
    fig, ax = plt.subplots()

    clusters = []
    for i in range(C.shape[0]):
        clusters.append([(0, 0)])

    for i in range(X.shape[0]):
#        for j in range(W.shape[0]):
        xc = np.argmax(W[i])
        xs = (X[i, 0], X[i, 1])
#        if clusters[xc] == 0:
#           clusters[xc][0] = xs[0]
#           clusters[xc][1] = xs[1]
#        else:
        clusters[xc].append(xs)

    for i in range(len(clusters)):
        Xix, Xiy = zip(*clusters[i])
        plt.scatter(Xix, Xiy)

    plt.scatter(C[:, 0], C[:, 1], marker='+') # Centroids
    plt.savefig(fname)


def kestep(m, S):
    for i in range(m.shape[0]):
        sum = [0, 0]
        for j in range(1, len(S[i])):
            sum[0] += S[i][j][0]
            sum[1] += S[i][j][1]
        t =  (sum[0] / len(S[i]), sum[1] / len(S[i]))
        m[i] = t
    return m


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
        S = kmstep(x, m, k, S)

        # Recompute the centroid of each cluster
        m = kestep(m, S)

        if np.any(ml != 0.0):
            t = m - ml
            if max(t.max(), t.min(), key=abs) < MINDELTA:
                break
        ml = copy.deepcopy(m)

    sse = sumofsquares(S)

    return m, S, sse


def kmstep (x, m, k, S):
    for p in range(x.shape[0]):
        t = np.zeros((k))
        for i in range(k):
            t[i] = twonorm(x[p], m[i])
        s = np.argmin(t)
        xs = (x[p, 0], x[p, 1])
        S[s].append(xs)

    return S


def kplot(S, m, fname):
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


def twonorm(x, m, p):
    sum = 0
    for i in range(x.size):
        sum += pow(x[i] - m[i], 2)
    if p == 1:
        return math.sqrt(sum)
    else:
        return sum


if __name__ == '__main__':
    main()