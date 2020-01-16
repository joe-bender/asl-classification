import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# number of sample images to take from each category
NUM_SAMPLES = 3

# create data.npy file to store the entire dataset
data_dir = os.path.join('asl-alphabet', 'asl_alphabet_train', 'asl_alphabet_train')
Xs_list = []
ys_list = []

# category names are the names of the directories
categories = os.listdir(data_dir)

# create category-to-integer index
cat2idx = {cat: idx for idx, cat in enumerate(categories)}

# load images and labels into lists
# (one list for image inputs and one list for category labels)
for category in categories:
    cat_dir = os.path.join(data_dir, category)
    image_files = os.listdir(cat_dir)
    y_idx = cat2idx[category]
    
    for image_file in image_files[:NUM_SAMPLES]:
        path = os.path.join(data_dir, category, image_file)
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        Xs_list.append([gray])
        ys_list.append(y_idx)

# turn lists into numpy arrays
Xs = np.vstack(Xs_list)
ys = np.vstack(ys_list)

# split intro train and test sets
test_size = .1
Xs_train, Xs_test, ys_train, ys_test = train_test_split(Xs, ys, test_size=test_size)

# save dataset to file
np_dir = 'numpy_data'
train_dir = 'train'
test_dir = 'test'

# create data directories
os.mkdir(np_dir)
os.mkdir(os.path.join(np_dir, train_dir))
os.mkdir(os.path.join(np_dir, test_dir))

# save datasets
np.save(os.path.join(np_dir, train_dir, 'Xs'), Xs_train)
np.save(os.path.join(np_dir, test_dir, 'Xs'), Xs_test)
np.save(os.path.join(np_dir, train_dir, 'ys'), ys_train)
np.save(os.path.join(np_dir, test_dir, 'ys'), ys_test)
