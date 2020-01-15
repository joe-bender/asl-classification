import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# number of sample images to take from each category
NUM_SAMPLES = 3

# create data.npy file to store the entire dataset
data_dir = os.path.join('asl-alphabet', 'asl_alphabet_train', 'asl_alphabet_train')
X_list = []
categories = os.listdir(data_dir)

for category in categories:
    cat_dir = os.path.join(data_dir, category)
    image_files = os.listdir(cat_dir)
    
    for image_file in image_files[:NUM_SAMPLES]:
        path = os.path.join(data_dir, category, image_file)
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        X_list.append([gray])

X = np.vstack(X_list)


# save dataset to file
np_dir = 'numpy_data'
os.mkdir(np_dir)
X_path = os.path.join(np_dir, 'X')
np.save(X_path, X)
