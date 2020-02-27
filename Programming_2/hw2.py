import numpy as np
import math

LABELPOS = 57
MINSTD = 0.0001
NFEATURES = 57


def main():
    print("Welcome to Programming #2")

    # 1. Create training and test set
    infile = 'spambase_data.csv'
    train, test = initData(infile)

    # 2. Create probabalistic model
#    p1 = np.sum(train[LABELPOS])
    pclass0 = 0
    pclass1 = 0

    class1 = np.sum(train[:, NFEATURES], axis=0, dtype=np.float64)
    pclass0 = (train.shape[0] - class1)/train.shape[0]
    pclass1 = class1/train.shape[0]
#    print("pclass0 = ", pclass0)
#    print("pclass1 = ", pclass1)
#    print("ptotal = ", pclass0 + pclass1)

    μclass0 = np.zeros(NFEATURES)
    μclass1 = np.zeros(NFEATURES)
    σclass0 = np.zeros(NFEATURES)
    σclass1 = np.zeros(NFEATURES)

    μclass1 = np.mean(train[:907, :NFEATURES], axis=0, dtype=np.float64)
    μclass0 = np.mean(train[907:, :NFEATURES], axis=0, dtype=np.float64)
    σclass1 = np.std(train[:907, :NFEATURES], axis=0, dtype=np.float64)
    σclass0 = np.std(train[907:, :NFEATURES], axis=0, dtype=np.float64)

    σclass1 = np.where(σclass1[:NFEATURES] == 0, MINSTD, σclass1)
    σclass0 = np.where(σclass0[:NFEATURES] == 0, MINSTD, σclass0)

    extrain = np.array([[3.0, 5.1, 1.0],
                       [4.1, 6.3, 1.0],
                       [7.2, 9.8, 1.0],
                       [2.0, 1.1, -1.0],
                       [4.1, 2.0, -1.0],
                       [8.1, 9.4, -1.0]])

    pexclass1 = 0.5
    pexclass0 = 0.5
    μexclass1 = np.mean(extrain[:3, :2], axis=0, dtype=np.float64)
    μexclass0 = np.mean(extrain[3:, :2], axis=0, dtype=np.float64)
    σexclass1 = np.std(extrain[:3, :2], axis=0, dtype=np.float64)
    σexclass0 = np.std(extrain[3:, :2], axis=0, dtype=np.float64)

    extest = np.array([5.2, 6.3])
    exclass = np.zeros((2, 2))

    for i in range(extest.size):
        exclass[i, 0] = ndist(extest[i], μexclass1[i], σexclass1[i])
        exclass[i, 1] = ndist(extest[i], μexclass0[i], σexclass0[i])

    posex = pexclass1 * exclass[0, 0] * exclass[1, 0]
    negex = pexclass0 * exclass[0, 1] * exclass[1, 1]

    if posex > negex:
        print("Example class = 1")
    else:
        print("Example class = 0")

    print()

#    print("p0 = ", p0)
#    print("p1 = ", p1)
#    print("p = ", p0 + p1)

    # 3. Run Bayesian learning model


def initData(infile):
    data = np.loadtxt(infile, delimiter=',')
    spam = data[:1813]
#    np.random.shuffle(spam)
#    print("spam = ",  spam)
#    print("spam shape =", spam.shape)
    nspam = data[1813:]
#    np.random.shuffle(nspam)
#    print("nspam = ",  nspam)
#    print("nspam shape =", nspam.shape)
    train = spam[:int(spam.shape[0] / 2) + 1, :]
    train = np.append(train, nspam[:int(nspam.shape[0] / 2), :], axis=0)
#    print("train = ", train)
#    print("train shape = ", train.shape)
    test = spam[int(spam.shape[0] / 2) + 1:, :]
    test = np.append(test, nspam[int(nspam.shape[0] / 2):, :], axis=0)
#    print("test = ", test)
#    print("test shape = ", test.shape)

    return train, test


def ndist(x, μ, σ):
    return (1 / (math.sqrt(2 * math.pi) * σ)) * (math.exp(-1 * pow(x - μ, 2) / (2 * pow(σ, 2))))


if __name__ == '__main__':
    main()