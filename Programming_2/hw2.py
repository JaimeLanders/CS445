import numpy as np

LABELPOS = 57

def main():
    print("Welcome to Programming #2")

    # 1. Create training and test set
    infile = 'spambase_data.csv'
    data = np.loadtxt(infile, delimiter=',')
    spam = data[:1813]
    np.random.shuffle(spam)
#    print("spam = ",  spam)
#    print("spam shape =", spam.shape)
    nspam = data[1813:]
    np.random.shuffle(nspam)
#    print("nspam = ",  nspam)
#    print("nspam shape =", nspam.shape)
    training = spam[:int(spam.shape[0] / 2) + 1, :]
    training = np.append(training, nspam[:int(nspam.shape[0] / 2), :], axis=0)
    print("training = ", training)
    print("training shape = ", training.shape)
    test = spam[int(spam.shape[0] / 2) + 1:, :]
    test = np.append(test, nspam[int(nspam.shape[0] / 2):, :], axis=0)
    print("test = ", test)
    print("test shape = ", test.shape)


if __name__ == '__main__':
    main()