
import cv2
import numpy as np
import os
from image import load_dataset, plot_image, normaliseImg, plot_many
from perturbation import gaussian_pixel_noise, get_random_gaussian_number, Gaussian_Blur, hsv_hue_noise_increase, occlusion, image_contrast_increase, image_brightness_increase, image_brightness_decrease, hsv_sat_noise_increase, image_contrast_decrease, recursive_blur
import matplotlib.pyplot as plt

std_base = [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]
std_norm = [element * 179 for element in std_base]
std_norm_sat = [element * 255 for element in std_base]

def perturb_test_set(folder_path): 
   
    for perturb_type in ['gaussian_pixel_noise', 'gaussian_blurring', 'image_contrast_increase', 'image_contrast_decrease', 'image_brightness_increase', 'image_brightness_decrease', 'hsv_hue_noise_increase', 'hsv_sat_noise_increase', 'occlusion']:
    
        if not os.path.exists(f"{folder_path}/{perturb_type}"):
            perturb_path = os.mkdir(f"{folder_path}/{perturb_type}")
        perturb_path = f"{folder_path}/{perturb_type}"


        for animal in ['CATS', 'DOGS']:
            print(animal)
            dataset = load_dataset(folder_path+'/'+animal)

            
            if perturb_type=='gaussian_pixel_noise':
            #apply gaussian pixel noise 
                for imgpath in dataset:
                    # print(imgpath)
                    img = cv2.imread(imgpath, -1)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    for std in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]:
                        modified = gaussian_pixel_noise(img, std)
                        name = imgpath.split("\\")[-1]
                        
                        std_path = f"{perturb_path}/{std}"
                        std_animal_path = f"{perturb_path}/{std}/{animal}"
                        if not os.path.exists(std_path):
                            os.mkdir(std_path)
                        if not os.path.exists(std_animal_path):
                            os.mkdir(std_animal_path)

                        cv2.imwrite(f"{std_animal_path}/{name}", modified)
            
            elif perturb_type=='gaussian_blurring':
                 
                 for imgpath in dataset:
                    # print(imgpath)
                    img = cv2.imread(imgpath, -1)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    for std in range(10):
                        modified = recursive_blur(img, std)
                        name = imgpath.split("\\")[-1]
                        
                        std_path = f"{perturb_path}/{std}"
                        std_animal_path = f"{perturb_path}/{std}/{animal}"
                        if not os.path.exists(std_path):
                            os.mkdir(std_path)
                        if not os.path.exists(std_animal_path):
                            os.mkdir(std_animal_path)

                        cv2.imwrite(f"{std_animal_path}/{name}", modified)
            
            elif perturb_type=='image_contrast_increase':
                for imgpath in dataset:
                    # print(imgpath)
                    img = cv2.imread(imgpath, -1)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    for std in [1.0, 1.03, 1.06, 1.09, 1.12, 1.15, 1.18, 1.21, 1.24, 1.27]:
                        modified = image_contrast_increase(img, std)
                        name = imgpath.split("\\")[-1]
                        
                        std_path = f"{perturb_path}/{std}"
                        std_animal_path = f"{perturb_path}/{std}/{animal}"
                        if not os.path.exists(std_path):
                            os.mkdir(std_path)
                        if not os.path.exists(std_animal_path):
                            os.mkdir(std_animal_path)

                        cv2.imwrite(f"{std_animal_path}/{name}", modified)

            elif perturb_type=='image_contrast_decrease':
                   for imgpath in dataset:
                    # print(imgpath)
                    img = cv2.imread(imgpath, -1)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    for std in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
                        modified = image_contrast_decrease(img, std)
                        name = imgpath.split("\\")[-1]
                        
                        std_path = f"{perturb_path}/{std}"
                        std_animal_path = f"{perturb_path}/{std}/{animal}"
                        if not os.path.exists(std_path):
                            os.mkdir(std_path)
                        if not os.path.exists(std_animal_path):
                            os.mkdir(std_animal_path)

                        cv2.imwrite(f"{std_animal_path}/{name}", modified)

            elif perturb_type=='image_brightness_increase':
                for imgpath in dataset:
                    # print(imgpath)
                    img = cv2.imread(imgpath, -1)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    for std in [0,5,10,15,20,25,30,35,40,45]:
                        modified = image_brightness_increase(img, std)
                        name = imgpath.split("\\")[-1]
                        
                        std_path = f"{perturb_path}/{std}"
                        std_animal_path = f"{perturb_path}/{std}/{animal}"
                        if not os.path.exists(std_path):
                            os.mkdir(std_path)
                        if not os.path.exists(std_animal_path):
                            os.mkdir(std_animal_path)

                        cv2.imwrite(f"{std_animal_path}/{name}", modified)

            elif perturb_type=='image_brightness_decrease':
                for imgpath in dataset:
                    # print(imgpath)
                    img = cv2.imread(imgpath, -1)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    for std in [0,5,10,15,20,25,30,35,40,45]:
                        modified = image_brightness_decrease(img, std)
                        name = imgpath.split("\\")[-1]
                        
                        std_path = f"{perturb_path}/{std}"
                        std_animal_path = f"{perturb_path}/{std}/{animal}"
                        if not os.path.exists(std_path):
                            os.mkdir(std_path)
                        if not os.path.exists(std_animal_path):
                            os.mkdir(std_animal_path)

                        cv2.imwrite(f"{std_animal_path}/{name}", modified)

            elif perturb_type=='hsv_hue_noise_increase':
                for imgpath in dataset:
                    # print(imgpath)
                    img = cv2.imread(imgpath, -1)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    
                    for std in std_norm:
                        modified = hsv_hue_noise_increase(img, std)
                        name = imgpath.split("\\")[-1]
                        
                        std_path = f"{perturb_path}/{std}"
                        std_animal_path = f"{perturb_path}/{std}/{animal}"
                        if not os.path.exists(std_path):
                            os.mkdir(std_path)
                        if not os.path.exists(std_animal_path):
                            os.mkdir(std_animal_path)

                        cv2.imwrite(f"{std_animal_path}/{name}", modified)

            elif perturb_type=='hsv_sat_noise_increase':
                for imgpath in dataset:
                    # print(imgpath)
                    img = cv2.imread(imgpath, -1)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    
                    for std in std_norm_sat:
                        modified = hsv_sat_noise_increase(img, std)
                        name = imgpath.split("\\")[-1]
                        
                        std_path = f"{perturb_path}/{std}"
                        std_animal_path = f"{perturb_path}/{std}/{animal}"
                        if not os.path.exists(std_path):
                            os.mkdir(std_path)
                        if not os.path.exists(std_animal_path):
                            os.mkdir(std_animal_path)

                        cv2.imwrite(f"{std_animal_path}/{name}", modified)

            elif perturb_type=='occlusion':
                for imgpath in dataset:
                    # print(imgpath)
                    img = cv2.imread(imgpath, -1)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    
                    for std in [0,5,10,15,20,25,30,35,40,45]:
                        modified = occlusion(img, std)
                        name = imgpath.split("\\")[-1]
                        
                        std_path = f"{perturb_path}/{std}"
                        std_animal_path = f"{perturb_path}/{std}/{animal}"
                        if not os.path.exists(std_path):
                            os.mkdir(std_path)
                        if not os.path.exists(std_animal_path):
                            os.mkdir(std_animal_path)

                        cv2.imwrite(f"{std_animal_path}/{name}", modified)


            else:
                print('reached else')
                exit(0)
        


perturb_test_set('data/C')

