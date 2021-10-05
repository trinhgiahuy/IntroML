import numpy as np
from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt


def accuracy(y_pred, y_true):
    return np.sum(np.equal(y_pred, y_true)) / y_pred.shape[0]


def gaussian_parameter(X_train, label_vec):
    # Calculate prior for each class.
    prior = np.sum(label_vec, axis=0).T

    # Calculate mean in each class.
    # Ex: (3000xnum_class).T * (3000xnum_feature) = (num_class x num_feature)
    mean = np.matmul(label_vec.T, X_train) / np.sum(label_vec, axis=0).T

    # Calculate variance in each class.
    # Var[X] = E[X^2] - E[X]^2.
    mean_of_square = np.matmul(label_vec.T, np.square(X_train)) / np.sum(label_vec, axis=0).T
    sigma = np.sqrt(mean_of_square - np.square(mean))

    return prior, mean, sigma


def calculate_posterior(X_test, mean, sigma, prior):
    N_test = X_test.shape[0]
    num_class = class_label.shape[0]
    num_features = X_train.shape[1]

    # Gaussian probability density each example.
    pdf = np.ones((N_test, num_class))
    for i in range(num_features):
        # Gaussian probability density for each features and for all class.
        pdf_feature = norm.pdf(X_test[:, i].reshape(N_test, 1),
                               mean[:, i].reshape(1, num_class),
                               sigma[:, i].reshape(1, num_class))

        pdf = np.multiply(pdf, pdf_feature)

    # Calculate posterior values for all class.
    p_pred = (pdf * prior) / np.sum(pdf * prior, axis=1, keepdims=True)

    return p_pred


X_train = np.loadtxt('X_train.txt')
y_train = np.expand_dims(np.loadtxt('Y_train.txt'), 1)
X_test = np.loadtxt('X_test.txt')
y_test = np.expand_dims(np.loadtxt('Y_test.txt'), 1)


# Get all class label,
class_label = np.expand_dims(np.unique(y_train), 1)

# One-hot label.
label_vec = y_train == class_label.T


# Assume the data has Multivariate Gaussian distribution, calculate
# statistic information of data.
prior, mean, sigma = gaussian_parameter(X_train, label_vec)

# Make prediction.
p_pred = calculate_posterior(X_test, mean, sigma, prior)
y_pred = np.expand_dims(np.argmax(p_pred, axis=1), 1) + 1

# Print the accuracy on test set.
print("Accuracy for test data using Naive Bayes classifier:",
      accuracy(y_pred, y_test))


# Find row position of test data in test set that belong to class 1.
idx_test_class1 = np.where(y_test == 1.0)[0]

# Draw histogram of posteriori of test data from class 1.
plt.figure()
for i in range(class_label.shape[0]):
    plt.hist(p_pred[idx_test_class1, i], bins=np.arange(0, 1.1, 0.1),
             edgecolor='black', label=f"Class {i + 1}")
plt.legend()
plt.title("Posteriori values of all classes for test data in class 1")
plt.show()