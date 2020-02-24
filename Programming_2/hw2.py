import numpy as np

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

    μ = np.zeros(NFEATURES)
    σ = np.zeros(NFEATURES)

    μ = np.mean(train[:, :train.shape[0] - 1], axis=0, dtype=np.float64)
    σ = np.std(train[:, :train.shape[0] - 1], axis=0, dtype=np.float64)

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


if __name__ == '__main__':
    main()