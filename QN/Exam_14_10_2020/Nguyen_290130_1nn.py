import numpy as np


def get_data(path):
    file = open(path, 'r')
    lines = file.readlines()
    return np.genfromtxt(lines)


def accuracy(y_pred, y_ground):
    return np.sum(y_pred == y_ground) / y_ground.shape[0]


def euclidean_norm(X):
    return np.sqrt(np.sum(np.square(X), axis=2))


def one_NN_classifier(X_test, X_train, Y_train):
    test_shape = X_test.shape
    diff_matrix = X_test.reshape(test_shape[0], 1, test_shape[1]) - X_train
    norm_matrix = euclidean_norm(diff_matrix)
    ind_vector = np.argmin(norm_matrix, axis=1).reshape(X_test.shape[0], 1)

    return Y_train[ind_vector]


# Get train data.
X_train = np.genfromtxt('X_train.txt')
Y_train = np.genfromtxt('Y_train.txt')

# Get test data.
X_test = np.genfromtxt('X_test.txt')
Y_test = np.genfromtxt('Y_test.txt')
Y_test = Y_test.reshape(Y_test.shape[0], 1)

# Predict the label.
y_pred = one_NN_classifier(X_test, X_train, Y_train)

# Printout the result.
print("Accuracy using 1-NN classifier:", accuracy(y_pred, Y_test))
