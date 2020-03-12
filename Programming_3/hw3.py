import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import sys

MINDELTA = 0.1


def main():
    """Main entry point for the program, executes either k-Means or Fuzzy
        C-Means algorithms

    Uses command line arguments to determine mode (1 = k-Means, 2 = FCM) and
     hyper-parameters used to tune respective algorithm, executes the correct
     algorithm using parameters and creates a plot of the iteration over r
     runs with the lowest sum of squares error (sse).

    Params:
        None

    Returns:
        0 if successful
        -1 if unsuccessful due to incorrect initialization

    Raises:
        None

    """
    print("Welcome to Homework 3\n")
    infile = 'cluster_dataset.txt'

    data = np.loadtxt(infile)

#    fig, ax = plt.subplots()
#    plot(data, 'dplot.png')

    mode = int(sys.argv[1])

    if mode == 1:
        print("K-Means mode active")
        k = int(sys.argv[2])
        r = int(sys.argv[3])
        print(f'''k: {k}, r: {r}\n''')

        M = list(range(r))
        S = list(range(r))

        sse = np.zeros((r))
        for i in range(r):
            print(f'''Performing K-Means run {i + 1}''')
            M[i], S[i], sse[i] = kmeans(data, k)

        lsse = np.argmin(sse)
        print(f'''\nThe minimum SSE for the {r} runs is: {sse[lsse]}''')
        kplot(S[lsse], M[lsse], 'kplot.png')
    elif mode == 2:
        print("Fuzzy C-Means mode active")
        # Choose a number of clusters: c (a hyperparameter).
        c = int(sys.argv[2])
        m = int(sys.argv[3]) # Fuzzifier
        r = int(sys.argv[4])
        print(f'''c: {c}, m: {m}, r: {r}\n''')

        W = list(range(r))
        C = list(range(r))
        sse = np.zeros((r))
        for i in range(r):
            print(f'''Performing Fuzzy C-Means run {i + 1}''')
            W[i], C[i], sse[i] = cmeans(data, c, m)

        lsse = np.argmin(sse)
        print(f'''\nThe minimum SSE for the {r} runs is: {sse[lsse]}''')
        cplot(data, W[lsse], C[lsse], 'cplot.png')
    else:
        print('\nUsage: python hw3 mode hyperparam1 (hyperparam2) #runs, (i.e):'
              '       \nk-Means mode (1): python hw3 1 K r'
              '       \nFCM mode (2): python hw3 2 c m r')
        return -1

    return 0


def cestep(X, W, m, C):
    """Performs the E-Step of the CFM algorithm

    Uses the parameters to calculate the new membership grades (W) of the E-step
     for the FCM algorithm.

    Params:
        X: NumPy array of float values representing the data points
        W: NumPy array of float values representing the coefficients/membership
            grades for each data point to each cluster before E-step
        m: Integer representing the Fuzzifier hyper-parameter
        C: NumPy array of float values representing the centroids for each
            cluster

    Returns:
        W: NumPy array of float values representing the coefficients/membership
            grades for each data point to each cluster after E-step

    Raises:
        None

    """
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            sum = 0
            for k in range(C.shape[0]):
                sum += pow(twonorm(X[i], C[j], 1) /
                                  (twonorm(X[i], C[k], 1)), (2 / (m - 1)))

            W[i, j] = 1 / sum
    return W


def cmeans(X, c, m):
    """Executes the Fuzzy C-Means algorithm

    Uses the parameters to execute the Fuzzy C-Means algorithm and returns the
     membership grades (W), clusters (C) and sum of squares error (sse).

    Params:
        X: NumPy array of float values representing the data points
            grades for each data point to each cluster before E-step
        c: Integer representing the c hyper-parameter for the number of clusters
        m: Integer representing the Fuzzifier hyper-parameter

    Returns:
        W: NumPy array of float values representing the coefficients/membership
            grades for each data point to each cluster
        C: NumPy array of float values representing the centroids for each
            cluster
        sse: Float representing the sum of squares error for the current run of
              the FCM algorithm

    Raises:
        None

    """
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
    W = np.zeros((X.shape[0], c))
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i, j] = np.random.uniform(1, c)
        W[i] = W[i] / np.sum(W[i])

    C = np.zeros((c, X.shape[1])) # centroids
    cl = np.zeros((c))

    # Repeat until the algorithm has converged/stopping condition:
    sse = 0
    lsse = 0
    while True:
        # (I) Compute the centroid for each cluster (m-step).
        C = cmstep(X, W, m, c)

        # (II) For each data point, compute its coefficients/membership grades
        # for being in the clusters (e-step).
        W = cestep(X, W, m, C)

        sse = csse(X, W, C, m)
        if lsse != 0 and lsse - sse < MINDELTA:
            break
        lsse = sse

    return W, C, sse


def cmstep(X, W, m, c):
    """Performs the M-Step of the CFM algorithm

    Uses the parameters to calculate the new centroids (C) of the M-step for
     the FCM algorithm.

    Params:
        X: NumPy array of float values representing the data points
        W: NumPy array of float values representing the coefficients/membership
            grades for each data point to each cluster before M-step
        m: Integer representing the Fuzzifier hyper-parameter
        c: Integer representing the c hyper-parameter for the number of clusters
        C: NumPy array of float values representing the centroids for each
            cluster before the M-step

    Returns:
        C: NumPy array of float values representing the centroids for each
            cluster after the M-step

    Raises:
        None

    """
    C = np.zeros((c, X.shape[1]))
    for k in range(C.shape[0]):
        numsum = 0
        densum = 0
        for i in range(X.shape[0]):
            numsum += pow(W[i, k], m) * X[i]
            densum += pow(W[i, k], m)
        C[k] = numsum/densum

    return C


def cplot(X, W, C, fname):
    """Plots the data points and outputs to a file for the FCM algorithm

    Uses the data (X), membership grades (W) and Clusters (C) to create a plot
     for results of the FCM algorithm.

    Params:
        X: NumPy array of float values representing the data points
        W: NumPy array of float values representing the coefficients/membership
            grades for each data point to each cluster
        C: NumPy array of float values representing the centroids for each
            cluster
        fname: String representing the filename of the plot to output to

    Returns:
        None

    Raises:
        None

    """
    fig, ax = plt.subplots()

    clusters = []
    for i in range(C.shape[0]):
        cs = [(C[i][0].item(), C[i][1].item())]
        clusters.append(cs)

    for i in range(X.shape[0]):
        xc = np.argmax(W[i])
        xs = (X[i, 0], X[i, 1])
        clusters[xc].append(xs)

    for i in range(len(clusters)):
        Xix, Xiy = zip(*clusters[i])
        plt.scatter(Xix, Xiy)

    plt.scatter(C[:, 0], C[:, 1], marker='+') # Centroids
    plt.savefig(fname)


def csse(X, W, C, m):
    """Calculates and retunrs the sum of squares error for Fuzzy C-Means

    Takes the data (X), membership grades (W), centroids (C) and Fuzzifier (m)
     to calculate the sum of squares error for the FCM algorithm.

    Params:
        X: NumPy array of float values representing the data points
        W: NumPy array of float values representing the coefficients/membership
            grades for each data point to each cluster
        C: NumPy array of float values representing the centroids for each
            cluster
        m: Integer representing the Fuzzifier hyper-parameter

    Returns:
        sse: Float representing the sum of squares error for the current run of
              the FCM algorithm

    Raises:
        None

    """
    sse = 0
    sumx = 0
    sumy = 0
    for i in range(X.shape[0]):
        for j in range(C.shape[0]):
            sse += pow(W[i, j], m) * twonorm(X[i], C[j], 2)

    return sse


def kestep(M, S):
    """Performs the E-Step of the k-Means algorithm

    Uses the parameters to calculate the new centroids/means (M) of the E-step
     for the k-Means algorithm.

    Params:
        M: NumPy array of integers representing the means/centroids for the
         k-Means algorithm before the E-step
        S: NumPy array of float values representing the set of clusters and
         their respective data points

    Returns:
        M: NumPy array of integers representing the means/centroids for the
         k-Means algorithm after the E-step

    Raises:
        None

    """
    for i in range(M.shape[0]):
        sum = [0, 0]
        for j in range(1, len(S[i])):
            sum[0] += S[i][j][0]
            sum[1] += S[i][j][1]
        t =  (sum[0] / len(S[i]), sum[1] / len(S[i]))
        M[i] = t
    return M


def kmeans(X, k):
    """Executes the k-Means algorithm

    Uses the parameters to execute the k-Means algorithm and returns the means
    (M), set of clusters and data points (S) and the sum of squares error (SSE).

    Params:
        X: NumPy array of float values representing the data points
        k: Integer representing the k hyper-parameter for the number of clusters
         in the k-Means algorithm

    Returns:
        M: NumPy array of integers representing the means/centroids for the
         k-Means algorithm
        S: NumPy array of float values representing the set of clusters and
         their respective data points
        sse: Float representing the sum of squares error for the current run of
         the k-Means algorithm

    Raises:
        None

    """
    # 1. Select K points as initial centroids
    idx = np.random.choice(X.shape[0], size=k)
    M = X[idx, :]

    # 2. repeat until Centroid do not change:
    ml = np.zeros((k))
    S = []
    while True:
        # Form K clusters by assigning each point to its closest centroid
        S = []
        for i in range(k):
            t = [(M[i, 0], M[i, 1])]
            S.append(t)
        S = kmstep(X, M, k, S)

        # Recompute the centroid of each cluster
        M = kestep(M, S)

        if np.any(ml != 0.0):
            t = M - ml
            if max(t.max(), t.min(), key=abs) < MINDELTA:
                break
        ml = copy.deepcopy(M)

    sse = ksse(S)

    return M, S, sse


def kmstep (X, M, k, S):
    """Performs the M-Step of the k-Means algorithm

    Uses the parameters to calculate the new set of clusters and their data
     points of the M-step for the k-Means algorithm.

    Params:
        X: NumPy array of float values representing the data points
        M: NumPy array of integers representing the means/centroids for the
         k-Means algorithm
        k: Integer representing the k hyper-parameter for the number of clusters
         in the k-Means algorithm
        S: NumPy array of float values representing the set of clusters and
         their respective data points before the M-step

    Returns:
        S: NumPy array of float values representing the set of clusters and
         their respective data points after the M-step

    Raises:
        None

    """
    for p in range(X.shape[0]):
        t = np.zeros((k))
        for i in range(k):
            t[i] = twonorm(X[p], M[i], 2)
        s = np.argmin(t)
        xs = (X[p, 0], X[p, 1])
        S[s].append(xs)

    return S


def kplot(S, M, fname):
    """Plots the data points and outputs to a file

    Uses the clusters and their data points (S) and the set of centroids/means
    (M) to create a plot for results of the k-Means algorithm.

    Params:
        S: NumPy array of float values representing the set of clusters and
         their respective data points
        M: NumPy array of integers representing the means/centroids for the
         k-Means algorithm
        fname: String representing the filename of the plot to output to

    Returns:
        None

    Raises:
        None

    """
    fig, ax = plt.subplots()
    for i in range(len(S)):
        for j in range(len(S[i])):
            Six, Siy = zip(*S[i])
        plt.scatter(Six, Siy)
    plt.scatter(M[:, 0], M[:, 1], marker='+')
    plt.savefig(fname)


def ksse(S):
    """Calculates and returns the sum of squares error for k-Means

    Uses the set of clusters and their respective datapoints to calculate the
     sum of squares error for the k-Means algorithm.

    Params:
        X: NumPy array of float values representing the data points
        W: NumPy array of float values representing the coefficients/membership
            grades for each data point to each cluster
        C: NumPy array of float values representing the centroids for each
            cluster
        m: Integer representing the Fuzzifier hyper-parameter

    Returns:
        sse: Float representing the sum of squares error for the current run of
              the k-Means algorithm

    Raises:
        None

    """
    sse = 0
    sumx = 0
    sumy = 0
    for i in range(len(S)):
        for j in range(1, len(S[i])):
            sumx += pow(S[i][j][0] - S[i][0][0], 2)
            sumy += pow(S[i][j][1] - S[i][0][1], 2)

    sse = sumx + sumy
    return sse


def twonorm(X, M, p):
    """Calculates the L2 norm

    Uses the data points (X) and the centroid/means (M) to calculate the L2 norm
     to the pth power.

    Params:
        X: NumPy array of float values representing the data points
        M: NumPy array of integers representing the means/centroids for the
         k-Means/FCM algorithm
        p: Integer representing the power to raise the L2 norm to; if 1 then a
            square root is performed on the sum before returning, if 2 then just
            the sum is returned

    Returns:
        sum: Float representing the result of the L2 norm calculation

    Raises:
        None

    """
    sum = 0
    for i in range(X.size):
        sum += pow(X[i] - M[i], 2)
    if p == 1:
        return math.sqrt(sum)
    else:
        return sum


if __name__ == '__main__':
    main()