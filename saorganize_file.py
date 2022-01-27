import os
import glob
import numpy as np
from shutil import copy as cp


def split_set(folder_path, source_path):
    
    if not os.path.exists(folder_path+"/Train"):
        train_path = os.mkdir(folder_path+"/Train")
        train_path = folder_path+"/Train"
    else:
        train_path = folder_path+"/Train"
    
    if not os.path.exists(folder_path+"/Test"):
        test_path = os.mkdir(folder_path+"/Test")
        test_path = folder_path+"/Test"
    else:
        test_path = folder_path+"/Test"
    
    if not os.path.exists(folder_path+"/Validate"):
        validation_path = os.mkdir(folder_path+"/Validate")
        validation_path = folder_path+"/Validate"
    else:
        validation_path = folder_path+"/Validate"
    
    for directory in os.listdir(source_path):
        directory_path = os.path.join(source_path, directory)
        print(directory_path)

        if not os.path.exists(train_path+f"/{directory}"):
            os.mkdir(train_path+f"/{directory}")
        if not os.path.exists(test_path+f"/{directory}"):
            os.mkdir(test_path+f"/{directory}")
        if not os.path.exists(validation_path+f"/{directory}"):
            os.mkdir(validation_path+f"/{directory}")

   
        for imgpath in glob.glob(directory_path+'/*.jpg'):
            print(imgpath)
            
            random_number = np.random.random()

            if random_number>0.4:
                cp(imgpath, os.path.join(train_path,directory))
                print(f"copied {imgpath} to {train_path}/{directory}")
            elif random_number>0.2:
                cp(imgpath, os.path.join(test_path,directory))
                print(f"copied {imgpath} to {test_path}/{directory}")
            else:
                cp(imgpath, os.path.join(validation_path,directory))
                print(f"copied {imgpath} to {validation_path}/{directory}")
            


split_set(folder_path="/Users/salvatoreesposito/Downloads/dummy_faces", source_path="/Users/salvatoreesposito/Downloads/Dummy_faces_data")