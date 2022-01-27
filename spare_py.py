############ old code ########## 
    # hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # hsv_norm = normaliseImg(hsv)
    # h,s,v = cv2.split(hsv_norm)
    
    # _, randomnoise = get_random_gaussian_number(0, std, shape=[224,224])

    # huenoise = h+randomnoise

    # if np.max(huenoise)>1.0:
    #     print('hit if statement')
    #     for index, row in enumerate(huenoise):
    #         for undex, value in enumerate(row):
    #             if value>1.0:
    #                 huenoise[index][undex]=value-1.0

    # merged = cv2.merge([huenoise, s, v])
    # merged_float32 = np.float32(merged)
    # converted_back = cv2.cvtColor(merged_float32, cv2.COLOR_HSV2RGB)
    # return randomnoise, converted_back


    ## other code method
    # hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # # hsv_norm = normaliseImg(hsv)
    # h,s,v = cv2.split(hsv)
    # print(f"minimum value h = {np.min(h)}, max value h =  {np.max(h)}")
    # print(f"minimum value s = {np.min(s)}, max value s =  {np.max(s)}")
    # print(f"minimum value v = {np.min(v)}, max value v =  {np.max(v)}")
    # _, randomnoise = get_random_gaussian_number(0, std, shape=[224,224])

    # randomnoise = np.array(randomnoise)
    # s = np.array(s)

    # satnoise = s + randomnoise
    # print(f"minimum value randomnoise = {np.min(randomnoise)}, max value randomnoise =  {np.max(randomnoise)}")

    # print(f"minimum value satnoise = {np.min(satnoise)}, max value satnoise =  {np.max(satnoise)}")

    # uintsatnoise = satnoise.astype(np.uint8)
    # print(f"minimum value uintsatnoise = {np.min(uintsatnoise)}, max value uintsatnoise =  {np.max(uintsatnoise)}")
    # if np.max(satnoise)>255:
    #     print('hit if statement')
    #     for index, row in enumerate(huenoise):
    #         for undex, value in enumerate(row):
    #             if value>255:
    #                 huenoise[index][undex]=255
    #             elif value<0:     # instructions do not say to include this for this part
    #                 huenoise[index][undex]=0

    # satnoise = satnoise.astype(np.uint8)
    # print(f"shape of satnoise {satnoise.shape}")
    # print(f"minimum value satnoise = {np.min(satnoise)}, max value satnoise =  {np.max(satnoise)}")

    # merged = cv2.merge([h, satnoise, v])

    # converted_back = cv2.cvtColor(merged, cv2.COLOR_HSV2RGB)

    def gaussian_pixel_noise(img):
    for undex in range(len(img)):
        for index in range(len(img[undex])):
            print(index)
            # if np.array_equal(image[undex][index], (pick_color)):
            #     print("pixel changed from: ", image[undex][index], " to: ", np.array(
            #         [np.random.randint(0, 20), np.random.randint(0, 20), np.random.randint(0, 20), 255]))
            #     image[undex][index] = np.array([np.random.randint(
            #         0, 20), np.random.randint(0, 20), np.random.randint(0, 20), 255])

    # os.chdir('C:\\Users\\mkedz\\Documents\\Ordino\\Technical\\Slash2Esports\\Image_Processing\\trust_gaming_logos\\')
    # cv2.imwrite(savename, image)


    # def image_contrast_increase(img, intensity):
#     ones = np.ones((224,224,3), dtype=np.uint8)
#     contrast_increase = ones*intensity
#     increased = img*contrast_increase
#     return increased.astype(np.uint8)
# def contrast_increase_simple(img, intensity):
#     return np.uint8(img*intensity)


# def image_brightness_increase(img, intensity):
#     ones = np.ones((224,224,3), dtype=np.uint8)
#     brightness_increase = ones*intensity
#     increased = img+brightness_increase
#     return increased.astype(np.uint8)

# def image_brightness_decrease(img, intensity):
#     ones = np.ones((224,224,3), dtype=np.uint8)
#     brightness_increase = ones*intensity
#     decreased = np.subtract(img, brightness_increase)
#     return decreased.astype(np.uint8)