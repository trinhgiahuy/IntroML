import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.activations import softmax
import keras
import matplotlib.pyplot as plt

"""
The accuracy of 1-NN is about 30%.
The accuracy of Bayes-classifier with resize the image to 4x4 is about 40%.
The accuracy of Convolutional Neural Networks in this implementation is about
60% with 50 epoch.
"""



def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

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
X = X.reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1).astype("int32")

# Ground truth table for training image.
Y_train = np.equal(np.array([Y]).T, np.arange(0, 10)).astype("int32")


# Get test data.
test_datadict = unpickle('cifar-10-batches-py/test_batch')

# Get test image.
X_test = test_datadict["data"].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("int32")

# Ground truth table for test data.
# Shape 10000x10.
Y_test = np.equal(np.array([test_datadict["labels"]]).T, np.arange(0, 10)).astype("int32")


model = Sequential()

# Model architecture,
model.add(Conv2D(32, (3, 3), strides=2, activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())

# Output layer.
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='sigmoid'))


keras.optimizers.SGD(lr=0.01)
model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])

model.summary()


# X = np.random.rand(50000, 3072)
history = model.fit(X, Y_train, epochs=5, verbose=1,
                    validation_data=(X_test, Y_test))

plt.figure(1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

for i in range(X_test.shape[0]):
    # Show some images randomly.
    if np.random.rand() > 0.80:
        plt.figure(1)
        plt.clf()
        plt.imshow(X_test[i])

        # Predicted label.
        y_pred = np.argmax(model.predict(np.array([X_test[i]])), axis=1)[0]

        # Ground truth label.
        y_ground = label_names[test_datadict["labels"][i]]

        plt.suptitle(f"Image {i + 1} (Left) has predicted label {label_names[y_pred]} \n"
                     f"Ground truth label {y_ground}")
        plt.pause(1)


