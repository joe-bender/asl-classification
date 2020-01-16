import numpy as np
import random
from sklearn.model_selection import train_test_split

def get_datasets():
    Xs = np.load('numpy_data/Xs.npy')
    ys = np.load('numpy_data/ys.npy')
    # number of inputs and labels should match
    assert(len(Xs) == len(ys))

    # split dataset into train, validation, and test
    test_size = .1
    val_size = .2
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(Xs, ys, test_size=test_size)
    Xs_train, Xs_val, ys_train, ys_val = train_test_split(Xs_train, ys_train, test_size=val_size)

    return Xs_train, Xs_val, Xs_test, ys_train, ys_val, ys_test

# randomly draw batch_size number of samples from the given dataset
def get_batch(Xs, ys, batch_size=8):
    # number of inputs and labels should match
    assert(len(Xs) == len(ys))
    sample_idxs = random.sample(range(len(ys)), batch_size)

    Xs_batch = Xs[sample_idxs, :]
    ys_batch = ys[sample_idxs]
    return Xs_batch, ys_batch
