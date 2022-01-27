import os
import glob
import numpy as np
from shutil import copy as cp


def split_set(folder_path, source_path):
    if not os.path.exists(folder_path+"/A"):
        train_path = os.mkdir(folder_path+"/A")
        train_path = folder_path+"/A"
    else:
        train_path = folder_path+"/A"
    
    if not os.path.exists(folder_path+"/B"):
        test_path = os.mkdir(folder_path+"/B")
        test_path = folder_path+"/B"
    else:
        test_path = folder_path+"/B"
    
    if not os.path.exists(folder_path+"/C"):
        validation_path = os.mkdir(folder_path+"/C")
        validation_path = folder_path+"/C"
    else:
        validation_path = folder_path+"/C"
    
    
    for directory in os.listdir(source_path):
        directory_path = os.path.join(source_path, directory)
        print(directory_path)

        if not os.path.exists(train_path+f"/{directory}"):
            os.mkdir(train_path+f"/{directory}")
        if not os.path.exists(test_path+f"/{directory}"):
            os.mkdir(test_path+f"/{directory}")
        if not os.path.exists(validation_path+f"/{directory}"):
            os.mkdir(validation_path+f"/{directory}")

        animal_types = range(1, 13)
        for animal_type in animal_types:
            numb = 0
            for imgpath in glob.glob(directory_path+f'/*_{animal_type}_*.png'):
                print(imgpath)
                
                if numb%3==0:

                    cp(imgpath, os.path.join(train_path,directory))
                    print(f"copied {imgpath} to {train_path}/{directory}")
                elif numb%3==1:
                    cp(imgpath, os.path.join(test_path,directory))
                    print(f"copied {imgpath} to {test_path}/{directory}")
                elif numb%3==2:
                    cp(imgpath, os.path.join(validation_path,directory))
                    print(f"copied {imgpath} to {validation_path}/{directory}")
                numb+=1
            


split_set(folder_path=r"C:\Users\pc1\Documents\IVC\Image_and_Vision\data", source_path=r"C:\Users\pc1\Documents\IVC\Image_and_Vision\catdog")