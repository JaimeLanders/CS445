import numpy as np
import matplotlib.pyplot as plt
import math

def main():
    print("Welcome to Homework 3")
    infile = 'cluster_dataset.txt'

    data = np.loadtxt(infile)

    fig, ax = plt.subplots()
#    plot(data, 'dplot.png')

    k = 3
    r = 1

    kmeans(data, k, r)


def assignment(x, m, k, S):
    #    return 0
    for p in range(x.shape[0]):
        t = np.zeros((k))
        for i in range(k):
            #        S[i] = assignment(x, m[i], k)
            t[i] = twonorm(x[p], m[i])
        #                np.append(S[i], assignment(x[p], m[i], k))
        s = np.argmin(t)
        #            np.append(S[3], (x[p, 0], x[p, 1]), axis=1)
        xs = (x[p, 0], x[p, 1])
        #            S = np.append(S, xs )
        S[s].append(xs)

    return S


def kmeans(x, k, r):
    print('kmeans()')
    # 1. Select K points as initial centroids
    idx = np.random.choice(x.shape[0], size=k)
    m = x[idx, :]

#    S = np.empty([k, 1], dtype=tuple)
    S = []
    for i in range(k):
        t = [(m[i, 0], m[i, 1])]
#        S[i, 0] = t
        S.append(t)

    # 2. repeat until Centroid do not change:
    ml = np.zeros((k))
    it = 0
    while True:
        print(it)
        it += 1
        # Form K clusters by assigning each point to its closest centroid
        S = assignment(x, m, k, S)

        # Recompute the centroid of each cluster
        m = update(m, S)

        if np.any(ml != 0.0):
            t = m - ml
            if max(t.max(), t.min(), key=abs) < 0.01:
                plot(m, 'mplot')
                return 0
        ml = m
        print()

    print()
    return 0


def plot(x, fname):
    fig, ax = plt.subplots()
    plt.scatter(x[:, 0], x[:, 1])
    plt.savefig(fname)


def twonorm(x, m):
    sum = 0
    for i in range(x.size):
        sum += pow(x[i] - m[i], 2)

    return math.sqrt(sum)


def update(m, S):
    for i in range(m.shape[0]):
        sum = [0, 0]
        for j in range(len(S[i])):
            sum[0] += S[i][j][0]
            sum[1] += S[i][j][1]
        t =  (sum[0] / len(S[i]), sum[1] / len(S[i]))
        m[i] = t
    return m


if __name__ == '__main__':
    main()