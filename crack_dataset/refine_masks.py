import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
import os

def get_list(file_list, type):
    image_list = []
    for i in range(len(file_list)):
        current_file_abs_path = os.path.abspath(file_list[i])
        if current_file_abs_path.endswith(type):
            image_list.append(current_file_abs_path)
        else:
            pass

    return image_list

mask_dir='../crack_dataset/masks/'
new_dir='../crack_dataset/new_masks/'
listdir=os.listdir(mask_dir)

for idx, data in enumerate(listdir):
    img = cv2.imread(mask_dir+data)
    ret, new_img = cv2.threshold(np.array(img), 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(new_dir+data, new_img)