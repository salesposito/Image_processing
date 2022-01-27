
import h5py
import numpy as np
from PIL import Image
import io
from os import listdir
from os.path import isfile, join
import cv2
import matplotlib.pyplot as plt
import pathlib
from tensorflow.keras import layers
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
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from keras.regularizers import l2
import pickle
from itertools import zip_longest
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import datetime
import datetime
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Input
AUTOTUNE = tf.data.experimental.AUTOTUNE



def training_data ():
    batch_size = 3
    input_height = 512
    input_width = 512

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '/Users/salvatoreesposito/Downloads/dummy_faces/Train',
    label_mode ='binary',
    validation_split=None,
    subset=None,
    seed=123,
    image_size=(input_height, input_width),
    batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '/Users/salvatoreesposito/Downloads/dummy_faces/Validate',
    label_mode ='binary',
    validation_split=None,
    subset=None,
    seed=123,
    image_size=(input_height, input_width),
    batch_size=batch_size)


    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '/Users/salvatoreesposito/Downloads/dummy_faces/Test',
    label_mode ='binary',
    validation_split=None,
    subset=None,
    seed=123,
    image_size=(input_height, input_width),
    batch_size=batch_size)
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds


train_ds, val_ds,test_ds = training_data()


def resize_and_augment(train_ds, val_ds,test_ds):


    data_augmentation = tf.keras.Sequential([layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2)])

    normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
    # image_batch, labels_batch = next(iter(normalized_ds))

    return train_ds, val_ds,test_ds

train_ds,val_ds,test_ds = resize_and_augment(train_ds, val_ds, test_ds)


def InceptionV3_model():

    input_tensor = Input(shape=(512, 512, 3))


    model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)

    # x = data_augmentation

    x = model.output
    
    # add a global spatial average pooling layer

    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)

    x = layers.Dropout(0.5)(x)

    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(1, activation='sigmoid')(x)

    # this is the model we will train
    model = Model(inputs=model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    # for layer in model.layers:
    #     layer.trainable = False

    model.summary()

    return model

model = InceptionV3_model ()



def compile_fit_model(train_ds, val_ds,model):

    # timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # name = 'res_net_50-'+timestr #

    # checkpoint_path = "checkpoints/"+name+"/cp-{epoch:04d}.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path)
    # os.system('mkdir {}'.format(checkpoint_dir))

    # tensorboard_callback = TensorBoard(
    # log_dir='tensorboard_logs/'+name,
    # histogram_freq=1)

    model.compile(optimizer = RMSprop(lr=0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])
    history = model.fit(train_ds, epochs=10, validation_data=val_ds, batch_size=3)
    # model.save("ResNet50_model")
    return history

history = compile_fit_model(train_ds, val_ds,model)



def plot_loss():
    plt.plot(history.history['loss'],'r',label='training loss')
    plt.plot(history.history['val_loss'],label='validation loss')
    plt.xlabel('# epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
plot_loss()
    

def plot_acc():
    plt.plot(history.history['accuracy'],'r',label='training accuracy')
    plt.plot(history.history['val_accuracy'],label='validation accuracy')
    plt.xlabel('# epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
plot_acc()



def evaluate_model(test_ds):

    loss, acc = model.evaluate(test_ds)
    print("Accuracy", acc)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
evaluate_model(test_ds)






