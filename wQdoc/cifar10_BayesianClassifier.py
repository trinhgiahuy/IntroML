import pickle
import numpy as np
from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


def cifar10_color(X):
    return np.mean(X, (1, 2))


def cifar10_2x2_color(X):
    N = X.shape[0]
    return (X.reshape(X.shape[0], 2, 16, 2, 16, 3).mean(axis=(2, 4))).reshape(N, 2 * 2 * 3)


def cifar10_nxn_color(X, n):
    N = X.shape[0]
    return (X.reshape(X.shape[0], n, 32 // n, n, 32 // n, 3).mean(axis=(2, 4))).reshape(N, n * n * 3)


def cifar_10_naivebayes_learn(Xf, Y):
    labels = np.arange(0, 10).reshape(10, 1)
    labels_vectorize = np.equal(Y.T, labels)

    # Number of images in each class.
    num_img = np.sum(labels_vectorize, 1).reshape(10, 1)

    # Priors.
    # Example: size of 10x1.
    p = num_img / np.sum(labels_vectorize)

    # Mean.
    # Example: Size of 10x3.
    mu = np.matmul(labels_vectorize, Xf) / num_img

    # Standard deviation.
    # Var[X] = E[X^2] - E[X]^2.

    # Calculate first term.
    mean_of_Xsquare = np.matmul(labels_vectorize, np.square(Xf)) / num_img

    # Example: Size of 10x3.
    sigma = np.sqrt(mean_of_Xsquare - np.square(mu))

    return mu, sigma, p


def norm_pdf(x, mu, sigma):
    test_size = x.shape[0]
    return norm.pdf(x.reshape(test_size, 1), mu.reshape(1, 10), sigma.reshape(1, 10))


def cifar10_classifier_naivebayes(x, mu, sigma, p):

    # Gaussian probability for each color channel.
    pdf_red = norm_pdf(x[:, 0], mu[:, 0], sigma[:, 0])
    pdf_green = norm_pdf(x[:, 1], mu[:, 1], sigma[:, 1])
    pdf_blue = norm_pdf(x[:, 2], mu[:, 2], sigma[:, 2])

    pdf_rgb = pdf_red * pdf_green * pdf_blue

    # Probability density of an image belong to a class.
    # Example: Size of 10000x10.
    p_class = pdf_rgb * p.T

    # Sum of probability of over all class.
    # Example: Size of 10000x1.
    p_x = np.sum(p_class, 1, keepdims=True)

    # Predicting labels.
    # Example: Size of 10000x1.
    label_pred = np.argmax(p_class / p_x, axis=1)

    return label_pred.reshape(len(label_pred), 1)


def cifar_10_bayes_learn(Xf, Y):

    feature_size = Xf.shape[1]

    labels = np.arange(0, 10).reshape(10, 1)

    # Example: size of 10x50000.
    labels_vectorized = np.equal(Y.T, labels)

    # Number of images in each class.
    num_img = np.sum(labels_vectorized, 1).reshape(10, 1)

    # Mean.
    mu = np.matmul(labels_vectorized, Xf) / num_img

    # Covariance matrix.
    # Example: size of 10x3x3.
    cov = np.zeros((10, feature_size, feature_size))

    for i in range(10):

        # Covariance matrix of a column vector X:
        # Cov(X) = E[X^T * X] - muT * mu.

        # First term.
        X_class = np.array(Xf[labels_vectorized[i, :], :])

        # Second term.
        muyT_x_muy = np.matmul(mu[i, :].reshape(feature_size, 1), mu[i, :].reshape(1, feature_size))

        cov[i, :, :] = (np.matmul(X_class.T, X_class) / X_class.shape[0]) - muyT_x_muy

    # Priors.
    # Example: size of 10x1.
    p = num_img / np.sum(labels_vectorized)

    return mu, cov, p


def cifar10_classifier_bayes(x, mu, cov, p):
    test_size = x.shape[0]

    # Multivariate probability density of images for each class.
    # Example: Size of 10000x10.
    pdf_rgb = np.zeros((test_size, 10))
    for i in range(10):
        pdf_rgb[:, i] = multivariate_normal.pdf(x, mu[i, :], cov[i])

    # Probability density of an image belong to a class.
    # Example: Size of 10000x10.
    p_class = pdf_rgb * p.T

    # Sum of probability of over all class.
    # Example: Size of 10000x1.
    p_x = np.sum(p_class, 1, keepdims=True)

    # Predicting labels.
    # Example: Size of 10000x1.
    label_pred = np.argmax(p_class / p_x, axis=1)

    return label_pred.reshape(len(label_pred), 1)


# Evaluation.
def class_acc(pred, gt):
    return np.sum(np.equal(pred, gt)) / gt.shape[0]


# Get all training data into memory.
prefix_path = 'cifar-10-batches-py/data_batch_'
labeldict = unpickle('cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]


# Get training data.
X = []
Y = []
for i in range(1, 6):
    data_path = prefix_path + str(i)
    datadict = unpickle(data_path)
    X.extend(datadict["data"])
    Y.extend(datadict["labels"])

# Get training images.
X = np.array(X)
X = X.reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint32")

# Ground truth table for training image.
Y = np.array([Y]).T


# Get test data.
test_datadict = unpickle('cifar-10-batches-py/test_batch')

# Get test image.
X_test = test_datadict["data"].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("int32")

# Ground truth table for test data.
Y_test = np.array([test_datadict["labels"]]).T


# Part 1.

# Resize training examples.
Xf = cifar10_color(X)

# Resize test set.
Xf_test = cifar10_color(X_test)

# Training.
mu, sigma, p = cifar_10_naivebayes_learn(Xf, Y)

# Predicting.
y_pred_naivebayes = cifar10_classifier_naivebayes(Xf_test, mu, sigma, p)

# Show result.
print("Accuracy of Naive Bayes learning:", class_acc(y_pred_naivebayes, Y_test))


# Part 2.

# Training.
mu, cov, p = cifar_10_bayes_learn(Xf, Y)

# Predicting.
y_pred_bayes = cifar10_classifier_bayes(Xf_test, mu, cov, p)

# Print result.
print("Accuracy of Bayes learning:", class_acc(y_pred_bayes, Y_test))


# Part 3.

# Resize image to size of 2x2.

# Resize training examples.
Xf_2x2 = cifar10_2x2_color(X)

# Resize test set.
Xf_test_2x2 = cifar10_2x2_color(X_test)

# Training.
mu, cov, p = cifar_10_bayes_learn(Xf_2x2, Y)

# Predicting.
y_pred_2x2 = cifar10_classifier_bayes(Xf_test_2x2, mu, cov, p)

# Print result.
print("Accuracy of Bayes learning with resize image to 2x2:", class_acc(y_pred_2x2, Y_test))

# Resize image to size of 2x2, 4x4, 8x8, 16x16 and 32x32.
performance_bayes = []
for i in range(6):
    # Side length of resize image.
    n = 2 ** i

    # Resize training examples.
    Xf_train_nxn = cifar10_nxn_color(X, n)

    # Resize test set.
    Xf_test_nxn = cifar10_nxn_color(X_test, n)

    # Training.
    mu, cov, p = cifar_10_bayes_learn(Xf_train_nxn, Y)

    # Predicting.
    y_pred_nxn = cifar10_classifier_bayes(Xf_test_nxn, mu, cov, p)

    # Store results.
    performance_bayes.append(class_acc(y_pred_nxn, Y_test))


# Also do for the first approach.
performance_naivebayes = []

for i in range(6):
    # Side length of resize image.
    n = 2 ** i

    # Resize training examples.
    Xf_train_nxn = cifar10_nxn_color(X, n)

    # Resize test set.
    Xf_test_nxn = cifar10_nxn_color(X_test, n)

    # Training.
    mu, sigma, p = cifar_10_naivebayes_learn(Xf_train_nxn, Y)

    # Predicting.
    y_pred_nxn = cifar10_classifier_naivebayes(Xf_test_nxn, mu, sigma, p)

    # Store results.
    performance_naivebayes.append(class_acc(y_pred_nxn, Y_test))

# Draw performance depend on resize images size.
plt.figure(1)
plt.plot(2 ** np.arange(6), performance_naivebayes, label="Independent color channel")
plt.plot(2 ** np.arange(6), performance_bayes, label="Multivariate Gaussian approach")
plt.legend()
plt.xlabel("Resize image side length")
plt.ylabel("Accuracy")
plt.title("Performance of two approach")
plt.show()

