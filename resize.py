import numpy as np
import PIL
from PIL import Image
import glob
import cv2

def crop_image(img_path:str, crop_size:int):
    img = cv2.imread(img_path)
    w,h,_=img.shape
    if crop_size>w or crop_size>h:
        return img
    crop_image=[]
    for x in range(0,w-crop_size,crop_size):
        for y in range(0,h-crop_size,crop_size):
            crop_image.append(img[x:x+crop_size,y:y+crop_size])
    return crop_image


