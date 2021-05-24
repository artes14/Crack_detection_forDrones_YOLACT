import numpy as np
import PIL
from PIL import Image
import glob
import matplotlib.pyplot as plt
import cv2
import time
def crop_image(img_path:str, crop_size:int, mode=None):
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

    # green to gray
    resultimg = cv2.bitwise_and(~img,~img, mask=mask)
    result=img*mask
    ax1.imshow(img)
    ax2.imshow(result)
    plt.show()
    cv2.imwrite('data/eval/crack_9_rmg.jpg', result)

def grcut(img):
    hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # green thres
    lower_g=np.array([30, 100,10])
    higher_g = np.array([80,255,100])

    # mask of image where there's green
    mask = cv2.inRange(hsv, lower_g, higher_g)
    # for plotting
    fig, ax=plt.subplots(3)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    newmask=np.zeros(mask.shape[:2],np.float64)
    #mask = np.zeros(img.shape[:2],cv2.CV_8UC1)
    print(mask)
    newmask[mask==0]=0
    newmask[mask==255]=1
    #rect = (0, 0, 1960, 4032)
    # img=img, mask=?, rect=roi, bgdModel=?, itercount=?, mode=?
    mask, _, _=cv2.grabCut(img, newmask, None, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img_cut = img * mask2[:, :, np.newaxis]

    ax[0].imshow(img)
    ax[1].imshow(img_cut)
    ax[2].imshow(mask)
    plt.show()

def example(img):
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    # Step 1
    rect = (0, 00, img.shape[0], img.shape[1])
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    # Step 2
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_g = np.array([33, 25, 10])
    higher_g = np.array([80, 255, 255])
    green=cv2.inRange(hsv, lower_g, higher_g)
    newmask = cv2.blur(green, (60, 60), 0)
    mask[newmask == 0] = 1
    mask[newmask > 1] = 0
    cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    mask2 = np.where((mask == 0) | (mask == 2), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
    plt.imshow(img), plt.colorbar(), plt.show()
    cv2.imwrite('data/eval/crack_9_gcut.jpg', img)

def blur(img):
    last_time=time.time()
    fig, ax = plt.subplots(3)
    # green thres
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_g = np.array([33, 25, 10])
    higher_g = np.array([80, 255, 255])

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 100))
    kernel_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

    mask = cv2.inRange(hsv, lower_g, higher_g)
    mask=cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_o)
    mask = cv2.blur(mask, (100, 100), 10)
    mask = cv2.dilate(mask, kernel, 100)
    mask = cv2.blur(mask, (100, 100), 10)
    #cv2.dilate(mask, kernel, 100)
    green=cv2.bitwise_and(img,img, mask)
    green=cv2.erode(green, kernel,100)
    cv2.imwrite('data/eval3/crack_9_mdil.jpg', mask)
    #mask=cv2.blur(mask,(15,15),10)
    #mask2 = cv2.blur(mask, (60, 60), 0)
    mask2 = cv2.blur(mask,(50, 50),0)

    mask_rgb=img.copy()
    mask_rgb[:,:,0]=mask2
    mask_rgb[:, :, 1] = mask2
    mask_rgb[:, :, 2] = mask2
    blur=cv2.blur(img,(60,60),0)

    blur = cv2.blur(blur, (60, 60), 0)
    out=img.copy()
    out[mask>0]=blur[mask>0]
    cv2.imwrite('data/eval3/crack_9_blr.jpg', out)

    # morphology - dilation

    dilation=cv2.dilate(out, kernel_o, 100)
    dilation_mask=cv2.dilate(mask,kernel,100)
    out=img.copy()
    out[dilation_mask>0]=dilation[dilation_mask>0]
    cv2.imwrite('data/eval3/crack_9_blr_dil.jpg',out)

    # for delayed time
    cur_time = time.time()
    elapsed = cur_time - last_time
    print(elapsed)

def blur_o(img):
    last_time=time.time()
    #blur = cv2.bilateralFilter(img, 30, 75, 75)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # green thres
    lower_g = np.array([33, 25, 10])
    higher_g = np.array([80, 255, 255])

    # mask of image where there's green
    mask = cv2.inRange(hsv, lower_g, higher_g)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    mask=cv2.blur(mask,(15,15),10)
    #mask2 = cv2.blur(mask, (60, 60), 0)
    mask2 = cv2.blur(mask,(50, 50),0)
    mask2 = cv2.dilate(mask, kernel, 100)
    mask2 = cv2.blur(mask, (100, 100), 10)

    mask_rgb=img.copy()
    mask_rgb[:,:,0]=mask2
    mask_rgb[:, :, 1] = mask2
    mask_rgb[:, :, 2] = mask2
    #blur=cv2.blur(img,(60,60),0)

    blur = cv2.blur(img, (60, 60), 0)
    out=img.copy()
    out[mask2>0]=blur[mask2>0]
    cv2.imwrite('data/eval2/crack_9_rmg.jpg', out)

    # morphology - dilation

    dilation=cv2.erode(out, kernel, 100)
    dilation_mask=cv2.dilate(mask2,kernel,100)
    out=img.copy()
    out[dilation_mask>0]=dilation[dilation_mask>0]
    cv2.imwrite('data/eval2/crack_9_dil.jpg',out)
    # for delayed time
    cur_time = time.time()
    elapsed = cur_time - last_time
    print(elapsed)


blur_o(cv2.imread('data/eval/crack_9.jpg'))