from random import random,randint,sample	#Import random built-in modules
import pickle
import numpy as np
import matplotlib.pyplot as plt


# Referenced from cifar10_illustrate.py
def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


# Referenced from cifar10_illustrate.py
def illustrate_cifar10():
    datadict = unpickle('/home/trinh/ML/Week2/cifar-10-batches-py/test_batch')

    X = datadict["data"]
    Y = datadict["labels"]

    labeldict = unpickle('/home/trinh/ML/Week2/cifar-10-batches-py/batches.meta')
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



#2. CIFAR-10 - Evaluation
"""
 function class_acc(pred,gt) that computes the classification accuracy for predicted labels `pred`
 as compared to the ground truth labels gt
"""
def class_acc(pred, gt):
    return np.sum(np.equal(pred, gt)) / len(gt)



#3 CIFAR-10 - Random classifier.
def cifar10_classifer_random(x):
    return randint(0, 9)

# Evaluate random classifier.
def evaluate_random_classifier(datadict_test):
    x_t = datadict_test["data"]

    # Getting predicted label.
    y_predicted = [cifar10_classifer_random(data) for data in x_t]

    # Getting ground truth labels.
    y_ground_truth = datadict_test["labels"]
    
    # Return the classification accuracy between predict label and ground truth label
    return class_acc(y_predicted , y_ground_truth)

# Define image difference using Euclidean distance formula 
def euclidean_norm_in_square(p1, p2):
    return np.sum(np.subtract(p1, p2) ** 2, 1)		#Given axis=1

def consine_distance(a, b):
    return np.argmin(np.matmul(b, np.transpose(a)) / np.sum(np.square(b), 1))

#4 CIFAR-10 - 1NN classifier
def cifar10_classifier_1nn(x, trdata, trlabels):
    rand_input_data = sample(range(1, np.shape(trdata)[0]), np.shape(trdata)[0] // 3)
    calculated_distance = euclidean_norm_in_square(x, trdata[rand_input_data])
    nearest_image_index = np.argmin(calculated_distance)
     
    return trlabels[nearest_image_index], nearest_image_index
    
    
    
    
    
# Get test data for #3
datadict_test = unpickle('/home/trinh/ML/Week2/cifar-10-batches-py/test_batch')
# Test of part 3.
print("Accuracy using random classifer:", evaluate_random_classifier(datadict_test))


# Get all training data into struct for #4
prefix_path = '/home/trinh/ML/Week2/cifar-10-batches-py/data_batch_'
all_training_data = []
all_training_label = []

for i in range(1, 6):
    data_path = prefix_path + str(i)
    datadict = unpickle(data_path)
    all_training_data.extend(datadict["data"])
    all_training_label.extend(datadict["labels"])

all_training_data = np.array(all_training_data)
all_training_label = np.array(all_training_label)

# Getting predicted data labels using 1-NN classifier.
result = [cifar10_classifier_1nn(data, all_training_data, all_training_label) for data in datadict_test["data"]]
y_predicted  = [res[0] for res in result]
nearest_neighbor_index = [res[1] for res in result]

# Save result to a binary file `1_NN_predictor.npy` in numpy format
np.save('1_NN_predictor.npy', result)


# Ground truth table for test data.
gt_label = datadict_test["labels"]

# Evaluate 1-NN classifier using classification accuracy function implemented
print("Accuracy using 1-NN classifier:", class_acc(y_predicted , gt_label))




# Referenced from cifar10_illustrate.py
training_image = all_training_data.reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1).astype("int32")
test_image = datadict_test["data"].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("int32")

labeldict = unpickle('/home/trinh/ML/Week2/cifar-10-batches-py/batches.meta')
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

        plt.suptitle(f"Image {i + 1} (Left) has predicted label {label_names[y_predicted [i]]} \n"
                     f"Ground truth label {label_names[gt_label[i]]} \n"
                     f"Nearest image on the right")
        plt.pause(5)
