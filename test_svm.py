import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import pylab as pl
from sklearn.metrics import confusion_matrix, accuracy_score
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
import cv2
from tqdm import tqdm
import pickle
from itertools import zip_longest
from sklearn.svm import SVC
from sklearn.metrics import plot_confusion_matrix
from scipy.cluster.vq import kmeans,vq
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import joblib

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



CATEGORIES = ["CATS", "DOGS"]

def get_xtest_ytest(training_data):
    X = []
    y = []

    for features, label in training_data:
        X.append(features)
        y.append(label)

    # X = np.array(X).reshape(-1, 128, 128, 3)
    return X, y

def create_testing_data(path, CATEGORIES):
    training_data = []
    for category in CATEGORIES:
          # do normal and cleft

        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=normal 1=cleft
        
        folderpath = os.path.join(path, category)
        # print(folderpath)
        for img in tqdm(os.listdir(folderpath)):  # iterate over each image
            # print(img)
            # print(os.path.join(folderpath,img))
            img_array = cv2.imread(os.path.join(folderpath,img) , -1)  # convert to array
            # new_array = cv2.resize(img_array, (224, 224))  # resize to normalize data size
            training_data.append([img_array, class_num])  # add this to our training_data


    return training_data


batch_size = 32
input_height = 224
input_width = 224

#before
loaded_model = joblib.load('finalized_svmmodel.sav')



data = []

for perturb_type in os.listdir('test'):
    # print(perturb_type)
    perturb_type_path = os.path.join('test', perturb_type)
    for perturb_level in os.listdir(perturb_type_path):
        # print(perturb_level)
        test_path = os.path.join(perturb_type_path, perturb_level)
        # print(test_path)
        
        testing_data = create_testing_data(test_path, CATEGORIES)
        # print(testing_data)
        X_test, y_test = get_xtest_ytest(testing_data)
        
        
        X_test = np.array(X_test)
        print("X_test shape: = ", X_test.shape)
        nsamples, nx, ny, rgb = X_test.shape
        X_test = X_test.reshape((nsamples,nx*ny*rgb))
        print("X_test shape: = ", X_test.shape)
        
        scaler = StandardScaler()
        X_test = scaler.fit_transform(X_test)
        X_test = scaler.transform(X_test)

        min_max_scaler = preprocessing.MinMaxScaler()
        X_test = min_max_scaler.fit_transform(X_test)

        print(X_test, y_test)
        
        # score = loaded_model.score(X_test, y_test)

        y_pred = loaded_model.predict(X_test)
        # calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        # print('Model accuracy is: ', accuracy)
        print(accuracy)
        # print("Test loss:", score[0])
        # print("Test accuracy:", score[1])
        data.append([test_path, score[1]])

print(data)

with open('testing_results_svm.pkl', 'wb') as f:
    pickle.dump(data, f)