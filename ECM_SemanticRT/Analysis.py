import cv2
import os
import matplotlib.pyplot as plt
from toolbox import class_to_RGB
from PIL import Image
import numpy as np


cmap = [
            (0, 0, 0),          # 0: unlabelled
            (64, 0, 128),       # 1: person
            (64, 64, 0),        # 2: car_stop
            (0, 128, 192),      # 3: bike
            (192, 0, 0),        # 4: bicyclist
            (128, 128, 0),      # 5: motorcycle
            (64, 64, 128),      # 6: motorcyclist
            (0, 255, 0),        # 7: car
            (192, 128, 128),    # 8: tricycle
            (192, 64, 0),       # 9: traffic_light
            (128, 138, 135),    # 10: box
            (128, 54, 15),      # 11:pole
            (0, 0, 255),        # 12:curve
        ]

image_path = '/Users/muscle/Desktop/01-16-2619/origin_img'
label_path = '/Users/muscle/Desktop/01-16-2619/gray_img'

Img_list = os.listdir(image_path)
for i in Img_list:
    idx_img = os.path.join(image_path,i)
    idx_label = os.path.join(label_path, i[:-4]+'.png')

    image = cv2.imread(idx_label,0)
    aa = np.where(image < 13, 0, image)
    # color_map = class_to_RGB(aa, N=len(cmap), cmap=cmap)
    # print(image.max())

    # color_map = class_to_RGB(image, N=len(cmap), cmap=cmap)
    # predict = Image.fromarray(predict)
    # predict.save(os.path.join(save_path, sample['label_path'][0]))


    plt.figure()
    plt.imshow(aa)
    # plt.imshow(color_map)
    plt.show()