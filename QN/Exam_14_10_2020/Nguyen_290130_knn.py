import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def get_data(path):
    file = open(path, 'r')
    lines = file.readlines()
    return np.genfromtxt(lines)


def accuracy(y_pred, y_ground):
    return np.sum(y_pred == y_ground) / y_ground.shape[0]


def euclidean_norm(X):
    return np.sqrt(np.sum(np.square(X), axis=2))


def k_NN_classifier(X_test, X_train, Y_train, K):
    test_shape = X_test.shape
    diff_matrix = X_test.reshape(test_shape[0], 1, test_shape[1]) - X_train
    norm_matrix = euclidean_norm(diff_matrix)
    ind_vector = np.argsort(norm_matrix, axis=1)[:, :K]

    vote_class = np.array(stats.mode(Y_train[ind_vector], axis=1)[0])

    return vote_class


# Get train data.
X_train = np.genfromtxt('X_train.txt')
Y_train = np.genfromtxt('Y_train.txt')

# Get test data.
X_test = np.genfromtxt('X_test.txt')
Y_test = np.genfromtxt('Y_test.txt')
Y_test = Y_test.reshape(Y_test.shape[0], 1)

k_set = [1, 2, 3, 5, 10, 20]
accuracy_set = []
for k in k_set:
    # Predict the label.
    y_pred = k_NN_classifier(X_test, X_train, Y_train, k)

    # Get the results.
    accuracy_set.append(accuracy(y_pred, Y_test))

plt.figure()
plt.plot(k_set, accuracy_set)
plt.show()
