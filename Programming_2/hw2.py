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

    μclass0 = np.zeros(NFEATURES)
    μclass1 = np.zeros(NFEATURES)
    σclass0 = np.zeros(NFEATURES)
    σclass1 = np.zeros(NFEATURES)

    μclass0 = np.mean(train[:907, :NFEATURES], axis=0, dtype=np.float64)
    μclass1 = np.mean(train[907:, :NFEATURES], axis=0, dtype=np.float64)
    σclass0 = np.std(train[:907, :NFEATURES], axis=0, dtype=np.float64)
    σclass1 = np.std(train[907:, :NFEATURES], axis=0, dtype=np.float64)

    σclass0 = np.where(σclass0[:NFEATURES] == 0, MINSTD, σclass0)
    σclass1 = np.where(σclass1[:NFEATURES] == 0, MINSTD, σclass1)

    # 3. Run Bayesian learning model
    tclass = np.zeros((test.shape[0], NFEATURES, 2))
    aclass = np.ones((test.shape[0], 2))
    aclass[:, 0] *= math.log2(pclass0)
    aclass[:, 1] *= math.log2(pclass1)

    for j in range(test.shape[0]):
        for i in range(NFEATURES):
            tclass[j, i, 0] = ndist(test[j, i], μclass0[i], σclass0[i])
            tclass[j, i, 1] = ndist(test[j, i], μclass1[i], σclass1[i])
            aclass[j, 0] += math.log2(tclass[j, i, 0])
            aclass[j, 1] += math.log2(tclass[j, i, 1])

    fclass = np.zeros((test.shape[0]))
    for i in range(test.shape[0]):
        if aclass[j, 0] > aclass[j, 1]:
            fclass[0] = 1
        else:
            fclass[0] = 0

    accuracy = np.sum(fclass) / np.sum(test[:, NFEATURES])

    print("accuracy = ", accuracy)


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
#    np.random.shuffle(spam)
    nspam = data[1813:]
#    np.random.shuffle(nspam)
    train = spam[:int(spam.shape[0] / 2) + 1, :]
    train = np.append(train, nspam[:int(nspam.shape[0] / 2), :], axis=0)
    test = spam[int(spam.shape[0] / 2) + 1:, :]
    test = np.append(test, nspam[int(nspam.shape[0] / 2):, :], axis=0)

    return train, test


def ndist(x, μ, σ):
    return (1 / (math.sqrt(2 * math.pi) * σ)) * (math.exp(-1 * pow(x - μ, 2) / (2 * pow(σ, 2))))


if __name__ == '__main__':
    main()
#    example()