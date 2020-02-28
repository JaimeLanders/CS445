import numpy as np
import math

MINSTD = 0.0001
NFEATURES = 57


def main():
    print("Welcome to Programming #2")

    # 1. Create training and test set
    infile = 'spambase_data.csv'
    train, test = initData(infile)

    # 2. Create probabalistic model
    p, μ, σ = probmodel(train)

    # 3. Run Bayesian learning model
    aclass = nblm(train, p, μ, σ)
    fclass = classify(test, aclass)

    accuracy = np.sum(fclass) / np.sum(test[:, NFEATURES])
    print("accuracy = ", accuracy)

    return 0


def classify(test, aclass):
    fclass = np.zeros((test.shape[0]))
    for i in range(test.shape[0]):
        if aclass[i, 0] > aclass[i, 1]:
            fclass[0] = 1
        else:
            fclass[0] = 0
    return fclass


def example():
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


def initData(infile):
    data = np.loadtxt(infile, delimiter=',')
    spam = data[:1813]
    nspam = data[1813:]
    train = spam[:int(spam.shape[0] / 2) + 1, :]
    train = np.append(train, nspam[:int(nspam.shape[0] / 2), :], axis=0)
    test = spam[int(spam.shape[0] / 2) + 1:, :]
    test = np.append(test, nspam[int(nspam.shape[0] / 2):, :], axis=0)

    return train, test


def nblm(test, p, μ, σ):
    tclass = np.zeros((test.shape[0], NFEATURES, 2))
    aclass = np.ones((test.shape[0], 2))
    aclass[:, 0] *= math.log2(p[0])
    aclass[:, 1] *= math.log2(p[1])

    for j in range(test.shape[0]):
        for i in range(NFEATURES):
            tclass[j, i, 0] = ndist(test[j, i], μ[i, 0], σ[i, 0])
            tclass[j, i, 1] = ndist(test[j, i], μ[i, 1], σ[i, 1])
            aclass[j, 0] += math.log2(tclass[j, i, 0])
            aclass[j, 1] += math.log2(tclass[j, i, 1])

    return aclass


def ndist(x, μ, σ):
    return (1 / (math.sqrt(2 * math.pi) * σ)) * (math.exp(-1 * pow(x - μ, 2) / (2 * pow(σ, 2))))


def probmodel(train):
    p = [0, 0]

    class1 = np.sum(train[:, NFEATURES], axis=0, dtype=np.float64)
    p[0] = (train.shape[0] - class1) / train.shape[0]
    p[1] = class1 / train.shape[0]

    μ = np.zeros((NFEATURES, 2))
    σ = np.zeros((NFEATURES, 2))

    μ[:, 0] = np.mean(train[:907, :NFEATURES], axis=0, dtype=np.float64)
    μ[:, 1] = np.mean(train[907:, :NFEATURES], axis=0, dtype=np.float64)
    σ[:, 0] = np.std(train[:907, :NFEATURES], axis=0, dtype=np.float64)
    σ[:, 1] = np.std(train[907:, :NFEATURES], axis=0, dtype=np.float64)

    σ[:, 0] = np.where(σ[:NFEATURES, 0] == 0, MINSTD, σ[:, 0])
    σ[:, 1] = np.where(σ[:NFEATURES, 1] == 0, MINSTD, σ[:, 1])

    return p, μ, σ


if __name__ == '__main__':
    main()
#    example()