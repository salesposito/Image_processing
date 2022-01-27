import cv2
import glob
path_to_train = 'train'
path_to_validation = 'validation'

for imgpath in glob.glob(path_to_train):
    print(imgpath)


from random import shuffle



