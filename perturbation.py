import numpy as np
import cv2
from image import normaliseImg

def recursive_blur(img, n):
    blurred = img
    for i in range(n):
        blurred = Gaussian_Blur(blurred)
    return blurred

def get_random_gaussian_number(mean, std, shape):
    return np.random.normal(mean, std, shape).astype(np.uint8), np.random.normal(mean, std, shape)

def gaussian_pixel_noise(img, std):
    _, number = get_random_gaussian_number(mean=0, std=std, shape=[224, 224, 3])
    img = np.array(img)
    number = np.array(number)
    modified = img+number
    if np.max(modified)>255 or np.min(modified)<0:
        print('hit if statement')
        for index, row in enumerate(modified):
            for undex, rgb in enumerate(row):
                for i, color in enumerate(rgb):
                    if color>255:
                        modified[index][undex][i]=255
                    if color<0:
                        modified[index][undex][i]=0
                 
    return np.array(modified, dtype='int')

def Gaussian_Blur(img, Intensity=1):
    ImageArray = img
    kernel = np.ones((3, 3), np.float32)
    kernel[0] = [1,2,1]
    kernel[1] = [2,4,2]
    kernel[2] = kernel[0]
    kernel = (kernel * Intensity)/16
    blurred = cv2.filter2D(ImageArray, -1, kernel)
    return blurred


def image_contrast_increase(img, intensity):
    ones = np.ones((224,224,3))
    contrast_increase = ones*intensity
    increased = np.multiply(img,contrast_increase)
    if np.max(increased)>255:
        print('hit if statement')
        for index, row in enumerate(increased):
            for undex, rgb in enumerate(row):
                for i, color in enumerate(rgb):
                    if color>255:
                        increased[index][undex][i]=255
                    # elif color<0:     # instructions do not say to include this for this part
                    #     increased[index][undex][i]=0
    
    return np.array(increased, dtype='int')


def image_contrast_decrease(img, intensity):
    ones = np.ones((224,224,3))
    contrast_decrease = ones*intensity
    decreased = np.multiply(img,contrast_decrease)
    if np.min(decreased)<0:
        print('hit if statement')
        for index, row in enumerate(decreased):
            for undex, rgb in enumerate(row):
                for i, color in enumerate(rgb):
                    if color<0:
                        decreased[index][undex][i]=0
                    # elif color<0:     # instructions do not say to include this for this part
                    #     increased[index][undex][i]=0
    
    return np.array(decreased, dtype='int')



def image_brightness_increase(img, intensity):
    ones = np.ones((224,224,3))
    brightness_increase = ones*intensity
    increased = img+brightness_increase
    if np.max(increased)>255:
        print('hit if statement')
        for index, row in enumerate(increased):
            for undex, rgb in enumerate(row):
                for i, color in enumerate(rgb):
                    if color>255:
                        increased[index][undex][i]=255
                    # elif color<0:     # instructions do not say to include this for this part
                    #     increased[index][undex][i]=0
    return np.array(increased, dtype='int')




def image_brightness_decrease(img, intensity):
    img = np.array(img)
    # print(img.shape)
    ones = np.ones((224,224,3))
    brightness_decrease = ones*intensity
    # print(brightness_decrease)
    # print(brightness_decrease.shape)
    decreased = img-brightness_decrease
    # decreased = cv2.subtract(img,brightness_decrease)
    if np.min(decreased)<0:
        print('hit if statement')
        for index, row in enumerate(decreased):
            for undex, rgb in enumerate(row):
                for i, color in enumerate(rgb):
                    if color<0:
                        decreased[index][undex][i]=0
                    # elif color<0:     # instructions do not say to include this for this part
                    #     increased[index][undex][i]=0

    return np.array(decreased, dtype='int')


def hsv_hue_noise_increase(img, std):

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h,s,v = cv2.split(hsv)

    _, randomnoise = get_random_gaussian_number(0, std, shape=[224,224])

    randomnoise = np.array(randomnoise)
    h = np.array(h)

    huenoise = h + randomnoise

    if np.max(huenoise)>179:
        print('hit if statement')
        for index, row in enumerate(huenoise):
            for undex, value in enumerate(row):
                if value>179:
                    huenoise[index][undex]=value-179
                # elif value<0:     # instructions do not say to include this for this part
                #     huenoise[index][undex]=0

    huenoise = huenoise.astype(np.uint8)
    # huenoise = np.array(huenoise, dtype='int')
    merged = cv2.merge([huenoise, s, v])
    converted_back = cv2.cvtColor(merged, cv2.COLOR_HSV2RGB)
    

    return converted_back

def hsv_sat_noise_increase(img, std):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    h,s,v = cv2.split(hsv)

    _, randomnoise = get_random_gaussian_number(0, std, shape=[224,224])

    randomnoise = np.array(randomnoise)
    s = np.array(s)
    print(f'max s = {np.max(s)}, min s = {np.min(s)}')

    satnoise = s + randomnoise
    
    if np.max(satnoise)>255 or np.min(satnoise)<0:
        print('hit if statement')
        for index, row in enumerate(satnoise):
            for undex, val in enumerate(row):
                # for i, color in enumerate(rgb):
                if val>255:
                    satnoise[index][undex]=255
                if val<0:
                        satnoise[index][undex]=0

    uintsatnoise = satnoise.astype(np.uint8)
    # uintsatnoise = np.array(satnoise, dtype='int')

    merged = cv2.merge([h, uintsatnoise, v])

    converted_back = cv2.cvtColor(merged, cv2.COLOR_HSV2RGB)
    
    return converted_back


def occlusion(img, edge_length):
    square = np.zeros((edge_length, edge_length, 3))
    random_x = np.random.randint(0, img.shape[0]-edge_length)
    # print(random_x)
    random_y = np.random.randint(0, img.shape[1]-edge_length)
    # print(random_y)
    img2 = img.copy()
    # print(square.shape[0], square.shape[1])
    img2[random_x:(random_x+square.shape[0]), random_y:(random_y+square.shape[1])] = square
    
    return img2