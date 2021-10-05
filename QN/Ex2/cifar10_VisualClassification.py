import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random, randint, sample


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


# Provided code.
def illustrate_cifar10():
    datadict = unpickle('cifar-10-batches-py/test_batch')

    X = datadict["data"]
    Y = datadict["labels"]

    labeldict = unpickle('cifar-10-batches-py/batches.meta')
    label_names = labeldict["label_names"]

    X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
    Y = np.array(Y)

    for i in range(X.shape[0]):
        # Show some images randomly
        if random() > 0.999:
            plt.figure(1);
            plt.clf()
            plt.imshow(X[i])
            plt.title(f"Image {i} label={label_names[Y[i]]} (num {Y[i]})")
            plt.pause(1)


# Evaluation.
def (pred, gt):
    return np.sum(np.equal(pred, gt)) / len(gt)


# Random classifier.
def cifar10_classifer_random(x):
    return randint(0, 9)


# Evaluate random classifier.
def evaluate_random_classifier(test_datadict):
    x_t = test_datadict["data"]

    # Predicted label.
    y_p = [cifar10_classifer_random(data) for data in x_t]

    # Ground truth labels.
    y_gt = test_datadict["labels"]

    return class_acc(y_p, y_gt)


# Define image difference.
def euclidean_norm_in_square(p1, p2):
    return np.sum(np.subtract(p1, p2) ** 2, 1)

def consine_distance(a, b):
    return np.argmin(np.matmul(b, np.transpose(a)) / np.sum(np.square(b), 1))

# 1-NN classifier.
def cifar10_classifier_1nn(x, trdata, trlabels):
    ind_rand = sample(range(1, np.shape(trdata)[0]), np.shape(trdata)[0] // 3)
    calculated_distance = euclidean_norm_in_square(x, trdata[ind_rand])
    nearest_image_index = np.argmin(calculated_distance)

    return trlabels[nearest_image_index], nearest_image_index


# Part 3.

# Get test data.
test_datadict = unpickle('cifar-10-batches-py/test_batch')

# Test of part 3.
print("Accuracy using random classifer:", evaluate_random_classifier(test_datadict))


# Part 4.

# Get all training data into memory.
prefix_path = 'cifar-10-batches-py/data_batch_'
all_training_data = []
all_training_label = []

for i in range(1, 6):
    data_path = prefix_path + str(i)
    datadict = unpickle(data_path)
    all_training_data.extend(datadict["data"])
    all_training_label.extend(datadict["labels"])

all_training_data = np.array(all_training_data)
all_training_label = np.array(all_training_label)

# Predicted data labels by 1-NN classifier.
result = [cifar10_classifier_1nn(data, all_training_data, all_training_label) for data in test_datadict["data"]]
y_p = [res[0] for res in result]
nearest_neighbor_index = [res[1] for res in result]

np.save('1_NN_predictor.npy', result)


# Ground truth table for test data.
gt_label = test_datadict["labels"]

# Evaluate 1-NN classifier.
print("Accuracy using 1-NN classifier:", class_acc(y_p, gt_label))

# Pick some images.
training_image = all_training_data.reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1).astype("int32")
test_image = test_datadict["data"].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("int32")

labeldict = unpickle('cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]

for i in range(test_image.shape[0]):
    # Show some images randomly
    if random() > 0.99:
        plt.figure(1)
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.imshow(test_image[i])

        plt.subplot(1, 2, 2)
        plt.imshow(training_image[nearest_neighbor_index[i]])

        plt.suptitle(f"Image {i + 1} (Left) has predicted label {label_names[y_p[i]]} \n"
                     f"Ground truth label {label_names[gt_label[i]]} \n"
                     f"Nearest image on the right")
        plt.pause(5)