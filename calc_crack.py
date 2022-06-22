import matplotlib.pyplot as plt
import math
import Crack
import cv2
import numpy as np
import colorsys
from scipy.stats import sem
import imgmod
from numpy import linalg
import time

def calc_width(x1, y1, x2, y2, d):
    width=20*d*math.tan(37.5*math.pi/180)*math.fabs(x1-x2)/5184  # AOV=75
    height=20*d*math.tan(30*math.pi/180)*math.fabs(y1-y2)/3888   # AOV=60
    print("diagonal=",math.sqrt(width*width+height*height))
    print("width=",width)
    print("height=",height)

def calc(x1, y1, x2, y2, d, real):
    p_w=math.fabs(x1-x2)
    p_h=math.fabs(y1-y2)
    #width=real*5184/p_w
    height = real * 3888 / p_h
    #p_l=math.sqrt(p_w * p_w + p_h * p_h)
    #AOV_w=2*math.atan(width/2/d)*180/math.pi
    AOV_h = 2 * math.atan(height / 2 / d) * 180 / math.pi
    print(AOV_h)

def otsu_thres( img):
    """Use OTSU's Threshold Binarization -
    first do gaussian filtering then Otsu's"""
    image = np.copy(img)
    blur = cv2.medianBlur(image, 3, 0)
    blur = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    ret, img_thres = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
    img_thres = cv2.bitwise_not(img_thres)
    # img_thres=cv2.erode(img_thres,np.ones((5,5),np.uint8))
    return ret, img_thres

ima=cv2.imread('data/crack_laser_result1/DJI_0030_139_frame.png', cv2.IMREAD_GRAYSCALE)
imge=cv2.imread('data/crack_laser_result1/DJI_0030_139.png')
_, thr=otsu_thres(imge)
crack=Crack.Crack(imge)
cv2.imshow('thresholding', thr)

print(thr.shape)
cv2.imshow('original image',imge)

ret, thr =cv2.threshold(thr, 0, 255, cv2.THRESH_OTSU)
# im=cv2.bitwise_not(im)
_r, ima =cv2.threshold(ima, 0, 255, cv2.THRESH_OTSU)
thin=crack.thinning(thr)
bitand_thin=cv2.bitwise_and(thin, ima)
bitand_mask = cv2.bitwise_and(thr, ima)
cv2.imshow('thin', bitand_thin)
print('bitand', bitand_thin.sum())
print('thin', thr.sum())

img=crack.img_gray.copy()
t=time.time()
f=plt.figure()
clamp = lambda n, minn, maxn: max(min(maxn, n), minn)

def hsv2rgb(h,s,v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))

number_of_colors = 30

def convolute_pixel_kernel(img_eval, thin):
    image = img_eval.copy()
    skeleton = thin.copy()
    (iH, iW) = image.shape[:2]
    (mH, mW) = skeleton.shape[:2]
    N = 50
    total_sk = skeleton.sum()
    print_iter = (total_sk/255/N).__floor__()
    if iH != mH and iW != mW:
        print('different image and mask size')
        return None
    arr_out = np.zeros((iH, iW))
    img_out = imge.copy()
    num=0
    # BorderTypes : BORDER_CONSTANT | BORDER_REPLICATE | BORDER_REFLECT | BORDER_WRAP | BORDER_TRANSPARENT | BORDER_ISOLATED
    for y in np.arange(0, iH):
        for x in np.arange(0, iW):
            if skeleton[y, x] == 0:
                continue
            ii, i = 1.0, 0
            while ii>0.99:
                i += 1
                kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (i * 2 - 1, i * 2 - 1))
                (kH, kW) = kern.shape[:2]
                b = (kW - 1) // 2
                p = np.sum(kern)
                img_pad = cv2.copyMakeBorder(image, b, b, b, b, borderType=cv2.BORDER_REPLICATE)
                roi = img_pad[y:y + 2*b + 1, x:x + 2*b + 1]
                k = np.bitwise_and(roi, kern).sum()
                ii = k / p
            i-=1
            if i<2:
                arr_out[y, x]=0
            else:
                arr_out[y, x]=(i-1)*2-1
            # draw ellipse on original imageA
            if num%print_iter==0:
                color = hsv2rgb(clamp((number_of_colors-i)/number_of_colors, 0, 1),1,1)
                img_out = cv2.ellipse(imge, (x,y), (i,i), 0, 0, 360, color)
            num += 1
    return arr_out, img_out

def convolute_pixel_avg(img_eval, thin):
    image = img_eval.copy()
    skeleton = thin.copy()
    (iH, iW) = image.shape[:2]
    (mH, mW) = skeleton.shape[:2]
    # N = 50
    # total_sk = skeleton.sum()
    # print_iter = (total_sk/255/N).__floor__()
    if iH != mH and iW != mW:
        print('different image and mask size')
        return None
    arr_out = np.zeros((iH, iW))
    img_out = imge.copy()
    sk_list=[]
    for y in np.arange(0, iH):
        for x in np.arange(0, iW):
            if skeleton[y, x] == 0:
                continue
            else:
                sk_list.append([x,y])
    x_std, y_std=np.std(sk_list, axis=0)

    # if horizontal crack
    if x_std > y_std:
        print("horizontal")
        for x, y in sk_list:
            i, j = x, y
            pix=0
            while i*j>0 and image[j,i]>0:
                pix+=1
                j+=1
            jmax=j
            i, j = x, y
            while i*j>0 and image[j,i]>0:
                pix+=1
                j-=1
            jmin=j
            pix=pix-1
            if pix>0:
                arr_out[y, x]=pix
                color = hsv2rgb(clamp((number_of_colors - pix) / number_of_colors, 0, 1), 1, 1)
                img_out=cv2.line(img_out, (x,jmax), (x,jmin), color, 1)

    # if vertical crack
    else:
        print("vertical")
        for x, y in sk_list:
            i, j = x, y
            pix=0
            while i*j>0 and image[j,i]>0:
                pix+=1
                i+=1
            imax=i
            i, j = x, y
            while i*j>0 and image[j,i]>0:
                pix+=1
                i-=1
            imin=i
            pix=pix-1
            if pix>0:
                arr_out[y, x]=pix
                color = hsv2rgb(clamp((number_of_colors - pix) / number_of_colors, 0, 1), 1, 1)
                img_out=cv2.line(img_out, (imax,y), (imin,y), color, 1)
    return arr_out, img_out

def show_crack_color(color_num):
    color_show = imge.copy()
    colshow = color_show[:color_num*10, :100]
    colshow = cv2.resize(colshow, dsize = (100,color_num*20))
    for i in range(color_num):
        # crack_width = crack.calc_pix(93,40,9216)*(2*i+1)
        crack_width = crack.calc_pix(93,40,9216)*(i)
        col = hsv2rgb(clamp((color_num-i)/color_num, 0, 1),1,1)
        colshow = cv2.rectangle(colshow, (0,i*20), (100,(i+1)*20), col, -1)
        colshow = cv2.putText(colshow, str(crack_width)[:4], (10,(i+1)*20-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
    return colshow

cv2.imshow('color map per width', show_crack_color(number_of_colors))
# cv2.imwrite('colormap2.png', show_crack_color(number_of_colors))

# result
# conv, result_im = convolute_pixel_kernel(bitand_mask, bitand_thin)
conv, result_im = convolute_pixel_avg(bitand_mask, bitand_thin)
filtered = conv[conv>0.0]
avg_pix = filtered.mean()
min_pix = filtered.min()
max_pix = filtered.max()
cv2.imshow('result',result_im)
t2=time.time()
t2=t2-t
print('time : ',t2)
print('mm per pixel :', crack.calc_pix(93,40,9216))
print('avg width : ', crack.calc_pix(93,40,9216)*avg_pix)
print('min width : ', crack.calc_pix(93,40,9216)*min_pix)
print('max width : ', crack.calc_pix(93,40,9216)*max_pix)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        copy_conv = param.copy()
        if copy_conv[y, x]>0:
            copy_img = result_im.copy()
            size = copy_conv[y, x]
            color = hsv2rgb(clamp((number_of_colors - size) / number_of_colors, 0, 1), 1, 1)
            # copy_img = cv2.ellipse(copy_img, (x,y), (int(size),int(size)), 0, 0, 360, color, thickness=2)
            copy_img = cv2.ellipse(copy_img, (x, y), (15,15), 0, 0, 360, color, thickness=1)
            crack_width = crack.calc_pix(93, 40, 9216) * (size)
            copy_img = cv2.putText(copy_img, str(crack_width)[:4], (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (color), 1, cv2.LINE_AA)
            cv2.imshow('result', copy_img)

cv2.setMouseCallback('result', mouse_callback, conv)
cv2.waitKey(0)
