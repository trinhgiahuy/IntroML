import pickle
import numpy as np
import matplotlib.pyplot as plt

"""
Description: Nodes is arrange in a square, where nodes in one end side is connected to
             the node on the opposite site.
"""

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


# Return index of neighbor of best matching unit.
def find_neighbor(ind, K):
    row_ind = ind // side
    col_ind = ind % side

    neighbor_ind = []

    # Right.
    if col_ind + 1 <= side - 1:
        neighbor_ind.append(ind + 1)
    else:
        # Turn to left end.
        neighbor_ind.append(row_ind * side)

    # Above.
    if row_ind - 1 >= 0:
        neighbor_ind.append(ind - side)
    else:
        # Turn to the bottom.
        neighbor_ind.append((side - 1) * side + col_ind)

    # Left.
    if col_ind - 1 >= 0:
        neighbor_ind.append(ind - 1)
    else:
        # Turn to right - end.
        neighbor_ind.append(row_ind * side + (side - 1))

    # Below.
    if row_ind + 1 <= side - 1:
        neighbor_ind.append(ind + side)
    else:
        # Turn to top.
        neighbor_ind.append(col_ind)

    return np.array(neighbor_ind).astype("uint8")


def consine_distance(a, b):
    return np.argmin(np.matmul(b, np.transpose(a)) / np.sum(np.square(b), 1))

def show_process(M):
    M_temp = M.reshape(K, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")

    # Create a board that show all node images.
    show_image = np.zeros((side * 32, side * 32, 3)).astype("uint8")
    for row in range(side):
        for col in range(side):
            show_image[(row * 32):(row + 1) * 32, col * 32:(col + 1) * 32] = M_temp[row * side + col]

    plt.figure(1)
    plt.imshow(show_image)
    plt.pause(0.0001)


# Repeat times.
R = 5

# Load pictures on test datasets and convert into vector form.
datadict = unpickle('cifar-10-batches-py/test_batch')

X = []
X.extend(datadict["data"])

# Get all training data into memory.
prefix_path = 'cifar-10-batches-py/data_batch_'

for i in range(1, 6):
    data_path = prefix_path + str(i)
    datadict = unpickle(data_path)
    X.extend(datadict["data"])

X = np.array(X).astype(float)

# Number of node on each side of the square.
side = 15

# Total number of node.
K = side * side

# Initialize K nodes.
M = np.random.randint(0, 255, (K, 3072)).astype(float)

# Choose alpha.
alpha = 0.4

# Update nodes.
for r in range(R):

    for i in range(np.shape(X)[0]):
        print("Round", r + 1, "- Image number", i + 1)

        # Find closest matching unit.
        # ind_min = np.argmin(np.sum(np.square(X[i] - M), 1))
        ind_min = consine_distance(X[i], M)

        # Adjust m_BMU for i.
        M[ind_min] += alpha * (X[i] - M[ind_min])

        # Adjust neighbor of m_BMU.
        neighbor_ind = find_neighbor(ind_min, K)
        for n_ind in neighbor_ind:
            M[n_ind] += 0.25 * alpha * (X[i] - M[n_ind])
            M[M < 0] = 0
            M[M > 255] = 255

        if i % 100 == 0:
            show_process(M)


# Create a board that show all node images.
show_process(M)
plt.show()