import numpy as np
import random
import os
from sklearn.model_selection import train_test_split

def get_datasets(train=True):
    if train:
        data_dir = 'train'
    else:
        data_dir = 'test'

    Xs = np.load(os.path.join('numpy_data', data_dir, 'Xs.npy'))
    ys = np.load(os.path.join('numpy_data', data_dir, 'ys.npy'))
    # number of inputs and labels should match
    assert(len(Xs) == len(ys))

    if train:
        # split dataset into train, validation, and test
        val_size = .2
        Xs_train, Xs_val, ys_train, ys_val = train_test_split(Xs, ys, test_size=val_size)
        return Xs_train, Xs_val, ys_train, ys_val
    else:
        return Xs, ys

# randomly draw batch_size number of samples from the given dataset
def get_batch(Xs, ys, batch_size=8):
    # number of inputs and labels should match
    assert(len(Xs) == len(ys))
    sample_idxs = random.sample(range(len(ys)), batch_size)

    Xs_batch = Xs[sample_idxs, :]
    ys_batch = ys[sample_idxs]
    return Xs_batch, ys_batch
