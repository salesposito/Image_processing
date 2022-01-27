import numpy as np
import io
from keras.utils import np_utils
from os import listdir
from os.path import isfile, join
import cv2
import matplotlib.pyplot as plt
import pathlib
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import keras
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import pickle
import dlib
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import datetime
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.model_selection import train_test_split
from itertools import zip_longest

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

def build_traindata():
    data_path='train_catdog'
    categories=os.listdir(data_path)
    labels=[i for i in range(len(categories))]
    label_dict=dict(zip(categories,labels)) #empty dictionary
    DATADIR = "train_catdog/"
    CATEGORIES = ["train_cats", "train_dogs"] 
    data=[]
    target=[]
    img_size=224
    for category in categories:
        path = os.path.join(DATADIR,category)
        for img in tqdm(os.listdir(path)):
                try:
                    img_array = cv2.imread(os.path.join(path,img))
                    # width = 224
                    # height = 224
                    # dim = (width, height)
                    # img = cv2.resize(img_array, dim, interpolation = cv2.INTER_AREA)  
                    # new_img = cv2.cvtColor(img_array, code=cv2.COLOR_BGR2RGB)          
                    data.append(img_array)
                    target.append(label_dict[category])

                except Exception as e:
                    print(str(e))
                    
    data=np.array(data)
    target=np.array(target)

    return data,target

data,target=build_traindata()

def training_split(data, target):

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25)
    print(X_train)
    # X_train = X_train / 255.0
    # X_test = X_test / 255.0
    # X_train = np.reshape(X_train,(X_train.shape[0],512,512,3))
    # X_test = np.reshape(X_test,(X_test.shape[0],512,512,3))
    # encode text category labels 
    # le = LabelEncoder() 
    # le.fit(y_train) 
    # y_train = le.transform(y_train) 
    # y_val = le.transform(y_val) 
    return X_train, X_test, y_train, y_test 

X_train, X_test, y_train, y_test = training_split(data, target)

def res_net_50 ():
    # The Convolutional Base of the Pre-Trained Model will be added as a Layer in this Model
    res_net_50_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))

    # output = res_net_50_model.layers[-1].output
    # output = keras.layers.Flatten()(output)
    
    # for layer in res_net_50_model.layers:
    #     layer.trainable = True

    model = Sequential()
    model.add(res_net_50_model)
    model.add(Flatten())
    model.add(Dense(512,activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.summary() 

    return model

model=res_net_50()


def compile_fit_model(X_train,y_train):

    # timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # name = 'res_net_50-'+timestr #
    # checkpoint_path = "checkpoints/"+name+"/cp-{epoch:04d}.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path)
    # os.system('mkdir {}'.format(checkpoint_dir))

    # tensorboard_callback = TensorBoard(
    # log_dir='tensorboard_logs/'+name,
    # histogram_freq=1)

    model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    history = model.fit(x=X_train,y=y_train,epochs=50,verbose=1,validation_split=0.2,batch_size=3)

    model.save("ResNet50_model")

    return history

history=compile_fit_model(X_train,y_train)


def plot_loss(history):
    plt.plot(history.history['loss'],'r',label='training loss')
    plt.plot(history.history['val_loss'],label='validation loss')
    plt.xlabel('# epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
plot_loss(history)
    
def plot_acc(history):
    plt.plot(history.history['accuracy'],'r',label='training accuracy')
    plt.plot(history.history['val_accuracy'],label='validation accuracy')
    plt.xlabel('# epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
plot_acc(history)


def evaluate_model():
    score = model.evaluate(X_test, y_test, batch_size=3,verbose=1)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
evaluate_model()

