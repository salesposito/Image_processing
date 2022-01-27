# import numpy as np
# from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, BatchNormalization, Activation, ZeroPadding2D, LeakyReLU, Conv2D, Conv2DTranspose
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.initializers import RandomNormal
# from tensorflow.keras.optimizers import Adam
# from tensorflow_addons.layers import InstanceNormalization
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.preprocessing.image import load_img
# import glob
# import cv2
# import matplotlib.pyplot as plt
# import imgaug as ia
# import imgaug.augmenters as iaa

import numpy as np
import time

import PIL.Image as Image
import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub

class CollectBatchStats(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []

  def on_train_batch_end(self, batch, logs=None):
    self.batch_losses.append(logs['loss'])
    self.batch_acc.append(logs['acc'])
    self.model.reset_metrics()


#props to ma boi dkersh aka Beadz aka David for making this sick class
class resNet50():

    def __init__(self, N_channels = 3, input_width = 224, input_height = 224, dataset_name = None):

        self.N_channels = N_channels
        self.input_width = input_width
        self.input_height = input_height
        self.dataset_name = dataset_name
        self.image_shape = (self.input_width, self.input_height)
        
        self.model_url = "https://tfhub.dev/tensorflow/resnet_50/classification/1"

        self.headless_model_url = "https://tfhub.dev/tensorflow/resnet_50/feature_vector/1"
        
        
        self.labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
        self.imagenet_labels = np.array(open(self.labels_path).read().splitlines())
        
        self.batch_size = 2

        # self.dataset_A = glob.glob("C:/Users/pc1/Documents/cyclegan/data/originals/5/*.TIF")
        # self.dataset_B = glob.glob("C:/Users/pc1/Documents/cyclegan/data/labels/*.png")

    def test_classification(self, imgpath):
        classifier = tf.keras.Sequential([hub.KerasLayer(self.model_url, input_shape=self.image_shape+(self.N_channels,))])
        img = Image.open(imgpath).resize(self.image_shape)
        img = np.array(img)/255.0
        print(img.shape)
        result = self.classifier.predict(img[np.newaxis, ...])
        print(result.shape)
        predicted_class = np.argmax(result[0], axis=-1)
        print(predicted_class)
        predicted_class_name = self.imagenet_labels[predicted_class]
        return predicted_class_name.title()

    def load_classifier(self):
        
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            'train',
            label_mode ='binary',
            validation_split=None,
            subset=None,
            seed=123,
            image_size=(self.input_height, self.input_width),
            batch_size=self.batch_size)
        
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            'validation',
            label_mode ='binary',
            validation_split=None,
            subset=None,
            seed=123,
            image_size=(self.input_height, self.input_width),
            batch_size=self.batch_size)

        
        print('train ds : ')
        print(train_ds)
        print(' val ds : ')
        print(val_ds)

        class_names = np.array(train_ds.class_names)
        # print(class_names)
        
        normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
        
        train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

        AUTOTUNE = tf.data.experimental.AUTOTUNE
        
        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


        print('train ds : ')
        print(train_ds)
        print(' val ds : ')
        print(val_ds)

        feature_extractor_layer = hub.KerasLayer(self.headless_model_url, input_shape=(self.input_height, self.input_width, self.N_channels), trainable=False)

        for image_batch, labels_batch in train_ds:
            print(image_batch.shape)
            print(labels_batch.shape)
            break
        
        feature_batch = feature_extractor_layer(image_batch)
        # print(feature_batch.shape)

        num_classes = len(class_names)

        model = tf.keras.Sequential([feature_extractor_layer, tf.keras.layers.Dense(num_classes)])
        model.summary()

        predictions = model(image_batch)
        # print(predictions.shape)

        return model, train_ds, val_ds

    def train(self):

        model, train_ds, val_ds = self.load_classifier()

        model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=['acc'])

        batch_stats_callback = CollectBatchStats()

        history = model.fit(train_ds, epochs=2, callbacks=[batch_stats_callback], validation_data=val_ds)

        plt.figure()
        plt.ylabel("Loss")
        plt.xlabel("Training Steps")
        plt.ylim([0,2])
        plt.plot(batch_stats_callback.batch_losses)

        plt.figure()
        plt.ylabel("Accuracy")
        plt.xlabel("Training Steps")
        plt.ylim([0,1])
        plt.plot(batch_stats_callback.batch_acc)
   