import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob 

def load_dataset(path):
    dataset = glob.glob(path+"/*.png")
    return dataset


def plot_image(img):
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.colorbar()
    plt.axis(False)
    plt.show()

def normaliseImg(img):
    img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    return img


def plot_many(images, title, title_values):
    cols = 5
    rows = 2

    fig = plt.figure(figsize=(20,8))
    for i in range(cols*rows):
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(np.squeeze(images[i]))
        plt.axis(False)
        plt.title(title+'='+str(title_values[i]))
            
    plt.show()


def plot_scatter(x, y, x_label, y_label, title):
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.show()