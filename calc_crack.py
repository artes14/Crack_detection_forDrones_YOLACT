import matplotlib.pyplot as plt
import math
import Crack
import cv2
import numpy as np
import colorsys
import os
from scipy.stats import sem
import imgmod
from numpy import linalg
import time


# camerainfo of previous {'angleW':70.5, 'pixelW':9248, 'angleH':55.9, 'pixelH':6936, 'distance':30}
camerainfo = {'angleW':70.5, 'pixelW':9248, 'angleH':55.9, 'pixelH':6936, 'distance':30}

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
    # blur = cv2.medianBlur(image, 3, 0)
    blur = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(blur,(5,5),0)
    # img_thres = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 20, 5)
    ret, img_thres = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
    img_thres = cv2.dilate(img_thres, np.ones((3,3),np.uint8))
    # img_thres=cv2.erode(img_thres,np.ones((3,3),np.uint8))
    ret=0
    img_thres = cv2.bitwise_not(img_thres)
    return ret, img_thres

# def new_thres(img):


def neighbours(x,y,image):
    "Return 8-neighbours of image point P1(x,y), in a clockwise order"
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],     # P2,P3,P4,P5
                img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]    # P6,P7,P8,P9

def transitions(neighbours):
    "No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence"
    n = neighbours + neighbours[0:1]      # P2, P3, ... , P8, P9, P2
    return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)

def zhangSuen(image):
    "the Zhang-Suen Thinning Algorithm"
    Image_Thinned = image.copy()  # deepcopy to protect the original image
    changing1 = changing2 = 1        #  the points to be removed (set as 0)
    while changing1 or changing2:   #  iterates until no further changes occur in the image
        # Step 1
        changing1 = []
        rows, columns = Image_Thinned.shape               # x for rows, y for columns
        for x in range(1, rows - 1):                     # No. of  rows
            for y in range(1, columns - 1):            # No. of columns
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1     and    # Condition 0: Point P1 in the object regions
                    2 <= sum(n) <= 6   and    # Condition 1: 2<= N(P1) <= 6
                    transitions(n) == 1 and    # Condition 2: S(P1)=1
                    P2 * P4 * P6 == 0  and    # Condition 3
                    P4 * P6 * P8 == 0):         # Condition 4
                    changing1.append((x,y))
        for x, y in changing1:
            Image_Thinned[x][y] = 0
        # Step 2
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1   and        # Condition 0
                    2 <= sum(n) <= 6  and       # Condition 1
                    transitions(n) == 1 and      # Condition 2
                    P2 * P4 * P8 == 0 and       # Condition 3
                    P2 * P6 * P8 == 0):            # Condition 4
                    changing2.append((x,y))
        for x, y in changing2:
            Image_Thinned[x][y] = 0
    return Image_Thinned

def thinning(mask):
    """thinning mask"""
    kernel=cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    img_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    img_th=cv2.ximgproc.thinning(img_open, img_open, thinningType=cv2.ximgproc.THINNING_GUOHALL)
    return img_th

def calc_pix(angle, dist, pix):
    """calculate mm per pixel,
    dist is in cm"""

    return 20*dist*math.fabs(math.tan(angle/2))/pix

def calc_crackwidth(img_i, img_m):
    t = time.time()
    img_origin = img_i.copy()
    img_gray = cv2.cvtColor(img_i, cv2.COLOR_BGR2GRAY)
    img_yolact = cv2.cvtColor(img_m, cv2.COLOR_BGR2GRAY)
    # make kernels in 3sizes s,m,l
    kernl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    kernm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    kerns = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    ''' this for CLAHE sequence
    img = cv2.GaussianBlur(img_gray, (3, 3), sigmaX=0, sigmaY=0)
    mask_dil = cv2.dilate(img_yolact, kernl)
    mask_dil = cv2.morphologyEx(mask_dil, cv2.MORPH_CLOSE, kernl)
    mask_ero = cv2.dilate(img_yolact, kerns)
    mask_ero = cv2.morphologyEx(mask_ero, cv2.MORPH_CLOSE, kernl)
    img_cla = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_cla = cv2.cvtColor(img_cla, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    img_cla[:, :, 0] = clahe.apply(img_cla[:, :, 0]) + 10
    img = cv2.cvtColor(img_cla, cv2.COLOR_YUV2BGR)
    masked_img = cv2.bitwise_and(img, img, mask=mask_dil)
    masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(masked_img, 0, 255, cv2.THRESH_OTSU)
    threshold = cv2.bitwise_not(threshold)
    result = cv2.bitwise_and(threshold, mask_ero)
    img_result_thin = thinning(result)
    '''
    ''' this for blackhat sequence'''
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), sigmaX=0, sigmaY=0)
    mask_dil = cv2.dilate(img_yolact, kernl)
    mask_dil = cv2.morphologyEx(mask_dil, cv2.MORPH_CLOSE, kernl)
    mask_ero = cv2.dilate(img_yolact, kerns)
    mask_ero = cv2.morphologyEx(mask_ero, cv2.MORPH_CLOSE, kernl)
    img_blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernm)
    result = cv2.subtract(img_gray, img_blackhat)
    result_and = cv2.bitwise_and(result, result, mask=mask_dil)
    _, result = cv2.threshold(result_and, 0, 255, cv2.THRESH_OTSU)
    result = cv2.bitwise_not(result)
    result = cv2.bitwise_and(result, mask_ero)
    img_result_thin = thinning(result)

    # _, img_thresh = otsu_thres(img_origin)
    # img_yolact = cv2.dilate(img_yolact, np.ones((7,7),np.uint8))
    # _, img_yolact_thres = cv2.threshold(img_yolact, 0, 255, cv2.THRESH_OTSU)
    # img_bitand = cv2.bitwise_and(img_thresh, img_yolact_thres)
    # img_bitand_thin =thinning(img_bitand)
    # mask_result = img_bitand.copy()

    # conv, img_result = convolute_pixel_grd(img_origin, img_bitand, img_bitand_thin)
    conv, img_result = convolute_pixel_DTplusGrad(img_origin, result, img_result_thin)
    filtered = conv[conv>0.0]
    if len(filtered)>0:
        avg_pix = filtered.mean()
        min_pix = filtered.min()
        max_pix = filtered.max()
    else:
        avg_pix, min_pix, max_pix = 0,0,0

    t2 = time.time()
    t2 = t2 - t
    print(t2, ' seconds elapsed')
    mmperpix = calc_pix(camerainfo['angleW'], camerainfo['distance'], camerainfo['pixelW'])
    return [mmperpix*min_pix, mmperpix*max_pix, mmperpix*avg_pix], img_result, result



# ima=cv2.imread('data/crack_laser_result1/DJI_0030_193_frame.png', cv2.IMREAD_GRAYSCALE)
# img_origin=cv2.imread('data/crack_laser_result1/DJI_0030_193.png')
# _, thr=otsu_thres(img_origin)
# crack=Crack.Crack(img_origin)
# cv2.imshow('thresholding', thr)
# cv2.imwrite('thresholding.png', thr)
#
# print(thr.shape)
# cv2.imshow('original image',img_origin)
#
# ret, thr =cv2.threshold(thr, 0, 255, cv2.THRESH_OTSU)
# # im=cv2.bitwise_not(im)
# _r, ima =cv2.threshold(ima, 0, 255, cv2.THRESH_OTSU)
# thin=crack.thinning(thr)
# bitand_thin=cv2.bitwise_and(thin, ima)
# bitand_mask = cv2.bitwise_and(thr, ima)
# cv2.imshow('thin', bitand_thin)
# cv2.imwrite('thin.png', bitand_thin)
# cv2.imwrite('bitand_mask.png', bitand_mask)
#
# print('bitand', bitand_thin.sum())
# print('thin', thr.sum())
#
# img=crack.img_gray.copy()
# t=time.time()
# f=plt.figure()
clamp = lambda n, minn, maxn: max(min(maxn, n), minn)

def hsv2rgb(h,s,v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))

number_of_colors = 30


def convolute_pixel_DT(img_origin, img_eval, thin):
    image = img_eval.copy()
    skeleton = thin.copy()
    (iH, iW) = image.shape[:2]
    DT = cv2.distanceTransform(image, cv2.DIST_L2, 5)
    arr_out = np.zeros((iH, iW))
    img_out = img_origin.copy()
    for y in np.arange(0, iH):
        for x in np.arange(0, iW):
            if skeleton[y, x] == 0:
                continue
            i = DT[y, x]
            color = hsv2rgb(clamp((number_of_colors - i) / number_of_colors, 0, 1), 1, 1)
            img_out = cv2.ellipse(img_origin, (x, y), (round(i), round(i)), 0, 0, 360, color)
            arr_out[y,x] = i*2-1
    return arr_out, img_out

def convolute_pixel_DTplusGrad(img_origin, img_eval, thin):
    # compute width according to skeleton slope
    image = img_eval.copy()
    skeleton = thin.copy()
    (iH, iW) = image.shape[:2]
    (mH, mW) = skeleton.shape[:2]
    # first DT then Gradient
    DT = cv2.distanceTransform(image, cv2.DIST_L2, 5)
    if iH != mH and iW != mW:
        print('different image and mask size')
        return None
    # outputs
    arr_out = np.zeros((iH, iW))
    img_out = img_origin.copy()
    # skeleton pixel locations
    sk_list=[]
    for y in np.arange(0, iH):
        for x in np.arange(0, iW):
            if skeleton[y, x] == 0:
                continue
            else:
                sk_list.append([x,y])
    pad=5
    skeleton_pad = cv2.copyMakeBorder(skeleton, pad,pad,pad,pad, cv2.BORDER_CONSTANT, value=[0])
    cnt=0
    for x, y in sk_list:
        cnt+=1
        if cnt%2!=0:
            continue
        if x<2 or y<2 or x>(iW-3) or y>(iH-3):
            continue
        # get roi
        roi = skeleton_pad[y:y+2*pad+1, x:x+2*pad+1]
        roi_list=[]
        for i in range(pad*2+1):
            for j in range(pad*2+1):
                if roi[j, i]==0:
                    continue
                else:
                    roi_list.append([i,j])
        # if there are enough points to calculate mse
        if len(roi_list)>5:
            roi_avg = [sum(i[0] for i in roi_list)/len(roi_list), sum(i[1] for i in roi_list)/len(roi_list)]
            roi_sub = np.subtract(roi_list, roi_avg)
            roi_t = (sum(i*j for i, j in roi_list)*len(roi_list) - sum(i[0] for i in roi_list)*sum(i[1] for i in roi_list))
            roi_b = (sum(i[0]*i[0] for i in roi_list)*len(roi_list) - sum(i[0] for i in roi_list)**2)
            if roi_t == 0 :
                roi_a = 100
            elif roi_b ==0:
                roi_a = 0
            else:
                roi_a = (-1)*roi_b/roi_t
            # roi_a, roi_b = np.polyfit(roi_x, roi_y, 1)
            i, j = x, y
            pix = 0
            add = 0
            if roi_a<-1 or roi_a>=1:
                while i * j > 0 and i<iW and j<iH and image[j, i] > 0:
                    pix += 1
                    add+=1
                    j += 1
                    if roi_a < -10 or roi_a > 10:
                        continue
                    i=round(i+add/roi_a)
                jmax=j
                p1=(i,jmax)
                add=0
                i, j = x, y
                while i*j>0 and i<iW and j<iH and image[j,i]>0:
                    pix+=1
                    add -= 1
                    j-=1
                    if roi_a < -10 or roi_a > 10:
                        continue
                    i=round(i+add/roi_a)
                jmin=j
                p2=(i,jmin)
            else:
                while i * j > 0 and i<iW and j<iH and image[j, i] > 0:
                    pix += 1
                    i += 1
                    add+=1
                    if roi_a > -0.1 or roi_a < 0.1:
                        continue
                    j = round(j + add * roi_a)
                imax=i
                p1=(imax,j)
                add=0
                i, j = x, y
                while i*j>0 and i<iW and j<iH and image[j,i]>0:
                    pix+=1
                    i-=1
                    add-=1
                    if roi_a > -0.1 or roi_a < 0.1:
                        continue
                    j = round(j - add * roi_a)
                imin=i
                p2=(imin,j)

            # for test
            # arr_out[y, x]=pix
            der_p = (p1[0]-p2[0]-2, p1[1]-p2[1]-2)
            dist_p = np.sqrt(der_p[0]*der_p[0]+der_p[1]*der_p[1])
            flag = False
            if dist_p>DT[y,x]*2:
                dist_p = DT[y,x]*2-1
                flag = True
            arr_out[y, x] = dist_p
            color = hsv2rgb(clamp((number_of_colors - dist_p) / number_of_colors, 0, 1), 1, 1)
            if flag:
                img_out = cv2.ellipse(img_origin, (x,y), (round(DT[y,x]),round(DT[y,x])), 0, 0, 360, color)
            else:
                img_out = cv2.line(img_out, p1, p2, color, 1)
        # m=x np.diff(roi_list, axis=0)

    return arr_out, img_out


def convolute_pixel_kernel(img_origin, img_eval, thin):
    image = img_eval.copy()
    skeleton = thin.copy()
    (iH, iW) = image.shape[:2]
    (mH, mW) = skeleton.shape[:2]
    N = 50
    total_sk = skeleton.sum()
    if total_sk/255<N:
        print_iter = 1
    else:
        print_iter = (total_sk/255/N).__floor__()
    if iH != mH and iW != mW:
        print('different image and mask size')
        return None
    arr_out = np.zeros((iH, iW))
    img_out = img_origin.copy()
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
                img_out = cv2.ellipse(img_origin, (x,y), (i,i), 0, 0, 360, color)
            num += 1
    return arr_out, img_out

def convolute_pixel_avg(img_origin, img_eval, thin):
    # conpute crack width horizontally or vertically only
    image = img_eval.copy()
    skeleton = thin.copy()
    (iH, iW) = image.shape[:2]
    (mH, mW) = skeleton.shape[:2]
    if iH != mH and iW != mW:
        print('different image and mask size')
        return None
    arr_out = np.zeros((iH, iW))
    img_out = img_origin.copy()
    x_avg, y_avg, pixel_num=0,0,0
    sk_list=[]
    for y in np.arange(0, iH):
        for x in np.arange(0, iW):
            if skeleton[y, x] == 0:
                continue
            else:
                x_avg+=x
                y_avg+=y
                pixel_num+=1
                sk_list.append((x,y))
    # use standard deviation for determining direction
    # x_std, y_std=np.std(sk_list, axis=0)
    # print(x_std, y_std)
    x_avg=x_avg/pixel_num
    y_avg=y_avg/pixel_num
    x_std, y_std=0,0
    for x, y in sk_list:
        x_std+=abs(x_avg-x)
        y_std+=abs(y_avg-y)
    print(x_std, y_std)

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

def convolute_pixel_grd(img_origin, img_eval, thin):
    # compute width according to skeleton slope
    image = img_eval.copy()
    skeleton = thin.copy()
    (iH, iW) = image.shape[:2]
    (mH, mW) = skeleton.shape[:2]
    if iH != mH and iW != mW:
        print('different image and mask size')
        return None
    # outputs
    arr_out = np.zeros((iH, iW))
    img_out = img_origin.copy()
    # skeleton pixel locations
    sk_list=[]
    for y in np.arange(0, iH):
        for x in np.arange(0, iW):
            if skeleton[y, x] == 0:
                continue
            else:
                sk_list.append([x,y])
    pad=3
    skeleton_pad = cv2.copyMakeBorder(skeleton, pad,pad,pad,pad, cv2.BORDER_CONSTANT, value=[0])
    cnt=0
    for x, y in sk_list:
        cnt+=1
        if cnt%2!=0:
            continue
        if x<2 or y<2 or x>(iW-3) or y>(iH-3):
            continue
        roi = skeleton_pad[y:y+2*pad+1, x:x+2*pad+1]
        roi_list=[]
        roi_x=[]
        roi_y=[]
        for i in range(pad*2+1):
            for j in range(pad*2+1):
                if roi[j, i]==0:
                    continue
                else:
                    roi_x.append(j)
                    roi_y.append(i)
                    roi_list.append([i,j])
        # if there are enough points to calculate mse
        if len(roi_x)>5:
            roi_avg = [sum(i[0] for i in roi_list)/len(roi_list), sum(i[1] for i in roi_list)/len(roi_list)]
            roi_sub = np.subtract(roi_list, roi_avg)
            roi_t = (sum(i*j for i, j in roi_list)*len(roi_list) - sum(i[0] for i in roi_list)*sum(i[1] for i in roi_list))
            roi_b = (sum(i[0]*i[0] for i in roi_list)*len(roi_list) - sum(i[0] for i in roi_list)**2)
            if roi_t == 0 :
                roi_a = 100
            elif roi_b ==0:
                roi_a = 0
            else:
                roi_a = (-1)*roi_b/roi_t
            # roi_a, roi_b = np.polyfit(roi_x, roi_y, 1)
            i, j = x, y
            pix = 0
            add = 0
            if roi_a<-1 or roi_a>=1:
                while i * j > 0 and i<iW and j<iH and image[j, i] > 0:
                    pix += 1
                    add+=1
                    j += 1
                    if roi_a < -10 or roi_a > 10:
                        continue
                    i=round(i+add/roi_a)
                jmax=j
                p1=(i,jmax)
                add=0
                i, j = x, y
                while i*j>0 and i<iW and j<iH and image[j,i]>0:
                    pix+=1
                    add -= 1
                    j-=1
                    if roi_a < -10 or roi_a > 10:
                        continue
                    i=round(i+add/roi_a)
                jmin=j
                p2=(i,jmin)
            else:
                while i * j > 0 and i<iW and j<iH and image[j, i] > 0:
                    pix += 1
                    i += 1
                    add+=1
                    if roi_a > -0.1 or roi_a < 0.1:
                        continue
                    j = round(j + add * roi_a)
                imax=i
                p1=(imax,j)
                add=0
                i, j = x, y
                while i*j>0 and i<iW and j<iH and image[j,i]>0:
                    pix+=1
                    i-=1
                    add-=1
                    if roi_a > -0.1 or roi_a < 0.1:
                        continue
                    j = round(j - add * roi_a)
                imin=i
                p2=(imin,j)

            # for test
            # arr_out[y, x]=pix
            der_p = (p1[0]-p2[0]-2, p1[1]-p2[1]-2)
            dist_p = np.sqrt(der_p[0]*der_p[0]+der_p[1]*der_p[1])
            arr_out[y, x] = dist_p
            color = hsv2rgb(clamp((number_of_colors - dist_p) / number_of_colors, 0, 1), 1, 1)
            img_out = cv2.line(img_out, p1, p2, color, 1)
        # m=x np.diff(roi_list, axis=0)

    return arr_out, img_out

def show_crack_color( color_num):
    # color_show = img_origin.copy()
    colshow = np.zeros((color_num*20, 100, 3), dtype=np.uint8)
    # colshow = color_show[:color_num*10, :100]
    # colshow = cv2.resize(colshow, dsize = (100,color_num*20))
    for i in range(color_num):
        crack_width = calc_pix(camerainfo['angleW'], camerainfo['distance'], camerainfo['pixelW'])*i
        # crack_width = crack.calc_pix(93,40,9216)*(i)
        col = hsv2rgb(clamp((color_num-i)/color_num, 0, 1),1,1)
        colshow = cv2.rectangle(colshow, (0,i*20), (100,(i+1)*20), col, -1)
        colshow = cv2.putText(colshow, str(crack_width)[:4], (10,(i+1)*20-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
    return colshow


# cv2.imshow('color map per width', show_crack_color(number_of_colors))
cv2.imwrite('colormap_A52.png', show_crack_color(number_of_colors))

# # result
# # conv, result_im = convolute_pixel_kernel(bitand_mask, bitand_thin)
# # conv, result_im = convolute_pixel_avg(bitand_mask, bitand_thin)
# conv, result_im = convolute_pixel_grd(bitand_mask, bitand_thin)
#
# filtered = conv[conv>0.0]
# avg_pix = filtered.mean()
# min_pix = filtered.min()
# max_pix = filtered.max()
# cv2.imshow('result',result_im)
# cv2.imwrite('result.png',result_im)
# t2=time.time()
# t2=t2-t
# print('time : ',t2)
# print('mm per pixel :', crack.calc_pix(93,40,9216))
# print('avg width : ', crack.calc_pix(93,40,9216)*avg_pix)
# print('min width : ', crack.calc_pix(93,40,9216)*min_pix)
# print('max width : ', crack.calc_pix(93,40,9216)*max_pix)
#
# def mouse_callback(event, x, y, flags, param):
#     if event == cv2.EVENT_MOUSEMOVE:
#         copy_conv = param.copy()
#         if copy_conv[y, x]>0:
#             copy_img = result_im.copy()
#             size = copy_conv[y, x]
#             color = hsv2rgb(clamp((number_of_colors - size) / number_of_colors, 0, 1), 1, 1)
#             copy_img = cv2.ellipse(copy_img, (x,y), (int(size/2),int(size/2)), 0, 0, 360, color, thickness=2)
#             # copy_img = cv2.ellipse(copy_img, (x, y), (15,15), 0, 0, 360, color, thickness=1)
#             crack_width = crack.calc_pix(93, 40, 9216) * (size)
#             copy_img = cv2.putText(copy_img, str(crack_width)[:4], (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (color), 1, cv2.LINE_AA)
#             cv2.imshow('result', copy_img)
#
# cv2.setMouseCallback('result', mouse_callback, conv)
# cv2.waitKey(0)
