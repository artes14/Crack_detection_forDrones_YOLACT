import numpy as np
import PIL
from PIL import Image
import glob
import matplotlib.pyplot as plt
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


def rm_g(img):
    hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # green thres
    lower_g=np.array([-10, 1,1])
    higher_g = np.array([80,255,100])

    # mask of image where there's green
    mask = cv2.inRange(hsv, lower_g, higher_g)
    fig, (ax1, ax2)=plt.subplots(2)

    # green to gray..?
    resultimg = cv2.bitwise_and(~img,~img, mask=mask)
    result=img+resultimg
    ax1.imshow(img)
    ax2.imshow(result)
    plt.show()
    cv2.imwrite('crack_9_rmg.jpg', result)
rm_g(cv2.imread('crack_9.jpg'))