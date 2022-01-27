import h5py
import numpy as np
from PIL import Image
import io
from os import listdir
from os.path import isfile, join
import cv2
import matplotlib.pyplot as plt
import pathlib
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import pickle
from itertools import zip_longest
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import datetime
import datetime
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

batch_size = 32
input_height = 224
input_width = 224

new_model = tf.keras.models.load_model('ResNet18_model_AB')
new_model.summary()

data = []
testing_folder = 'C'
for perturb_type in os.listdir(testing_folder)[2:10]:
    # print(perturb_type)
    perturb_type_path = os.path.join(testing_folder, perturb_type)
    for perturb_level in os.listdir(perturb_type_path):
        # print(perturb_level)
        test_path = os.path.join(perturb_type_path, perturb_level)
        print(test_path)
        
        
        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            test_path,
            label_mode ='binary',
            validation_split=None,
            subset=None,
            seed=123,
            image_size=(input_height, input_width),
            batch_size=batch_size)

        score = new_model.evaluate(test_ds)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])
        data.append([test_path, score[1]])

print(data)

with open('testing_results.pkl', 'wb') as f:
    pickle.dump(data, f)
