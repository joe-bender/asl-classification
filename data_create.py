import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# number of sample images to take from each category
NUM_SAMPLES = 3

# create data.npy file to store the entire dataset
data_dir = os.path.join('asl-alphabet', 'asl_alphabet_train', 'asl_alphabet_train')
Xs_list = []
ys_list = []

categories = os.listdir(data_dir)
# create category-to-integer index
cat2idx = {cat: idx for idx, cat in enumerate(categories)}

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

Xs = np.vstack(Xs_list)
ys = np.vstack(ys_list)

# save dataset to file
np_dir = 'numpy_data'
os.mkdir(np_dir)
Xs_path = os.path.join(np_dir, 'Xs')
np.save(Xs_path, Xs)
ys_path = os.path.join(np_dir, 'ys')
np.save(ys_path, ys)
