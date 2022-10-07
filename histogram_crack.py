import numpy as np
import glob
import matplotlib.pyplot as plt
import cv2
import time
import os
from pathlib import Path
import math
import Crack
import colorsys
from scipy.stats import sem
import imgmod

# 20220804_161427_29
# 20220804_161432_29
# 20220804_161427_45
# 20220804_161436_39  # 84
# 20220804_161436_40  # 82
# 20220804_161438_34  # 85
# 20220804_161427_58
# 20220804_161427_40
#



img_origin = cv2.imread('data/cloud_croptest/20220804_161427_29.png', cv2.IMREAD_GRAYSCALE)
(iH, iW) = img_origin.shape[:2]
# create a mask
mask = cv2.imread('data/cloud_croptest/20220804_161427_29_frame.png', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('tests/mask.png', mask)
kernl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
kerns = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

# circular kernel
mask_dil = cv2.dilate(mask, kernl)
mask_dil = cv2.morphologyEx(mask_dil, cv2.MORPH_CLOSE, kernl)
cv2.imwrite('tests/mask_dil.png', mask_dil)
mask_ero = cv2.dilate(mask, kerns)
mask_ero = cv2.morphologyEx(mask_ero, cv2.MORPH_CLOSE, kernl)
cv2.imwrite('tests/mask_ero.png', mask_ero)
thin=cv2.ximgproc.thinning(mask_dil, thinningType=cv2.ximgproc.THINNING_GUOHALL)
hist_origin = cv2.calcHist([img_origin],[0],None,[256],[0,256])
# gaussian blur original image
img = cv2.GaussianBlur(img_origin, (3,3), sigmaX=0, sigmaY=0)

# equalize image
# img = cv2.equalizeHist(img)

# normalize image
# img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

# canny edge original image?
# img = cv2.Canny(img, 180, 250)

# CLAHE
img_cla = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img_cla = cv2.cvtColor(img_cla, cv2.COLOR_BGR2YUV)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
img_cla[:,:,0] = clahe.apply(img_cla[:,:,0])+10
img = cv2.cvtColor(img_cla, cv2.COLOR_YUV2BGR)

# Background subtractor
# sub = cv2.createBackgroundSubtractorKNN(history=5)
# newmask = sub.apply(img)

masked_img = cv2.bitwise_and(img,img,mask = mask_dil)
masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
# Calculate histogram with mask and without mask
# Check third argument for mask
hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv2.calcHist([img],[0],masked_img,[256],[0,256])
result = cv2.morphologyEx(img_origin, cv2.MORPH_BLACKHAT, kernel)
result = cv2.subtract(img_origin, result)
hist_blackhat = cv2.calcHist([result],[0],None,[256],[0,256])
# result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
result = cv2.bitwise_and(result,result,mask = mask_dil)
_, result = cv2.threshold(result, 0, 255, cv2.THRESH_OTSU)
plt.subplot(231), plt.imshow(img_origin, 'gray')
# plt.subplot(222), plt.imshow(mask,'gray')
plt.subplot(233), plt.imshow(masked_img, 'gray')
sk_list = []
_, threshold = cv2.threshold(masked_img, 0, 255, cv2.THRESH_OTSU)
print(_)
# threshold = cv2.adaptiveThreshold(masked_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5,10)
# threshold = cv2.Canny(masked_img, 180, 250)
# threshold = cv2.Sobel(masked_img, cv2.CV_64F, dx=0, dy=1, ksize=3)
plt.subplot(232), plt.imshow(img, 'gray')
plt.subplot(234), # plt.imshow(threshold, 'gray')
plt.plot(hist_full), plt.plot(hist_mask), plt.plot(hist_blackhat)
threshold = cv2.bitwise_not(threshold)
threshold = cv2.bitwise_and(threshold, mask_ero)
plt.subplot(235), plt.imshow(threshold)
# result = cv2.bitwise_and(threshold, mask_ero)
result = cv2.bitwise_not(result)
cv2.imwrite('tests/not.png', result)
result = cv2.bitwise_and(result, mask_ero)
cv2.imwrite('tests/and.png', result)
# result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
plt.subplot(236), plt.imshow(result)
plt.show()

# for y in np.arange(0, iH):
#     for x in np.arange(0, iW):
#         if thin[y, x] == 0:
#             continue
#         else:
#             sk_list.append([x, y])
# pad = 3
# skeleton_pad = cv2.copyMakeBorder(thin, pad,pad,pad,pad, cv2.BORDER_CONSTANT, value=[0])
# cnt=0
#
# for x, y in sk_list:
#     cnt+=1
#     if cnt%2!=0:
#         continue
#     if x<2 or y<2 or x>(iW-3) or y>(iH-3):
#         continue
#     # get roi
#     roi = skeleton_pad[y:y+2*pad+1, x:x+2*pad+1]
#     roi_list=[]
#     for i in range(pad*2+1):
#         for j in range(pad*2+1):
#             if roi[j, i]==0:
#                 continue
#             else:
#                 roi_list.append([i,j])
#         # if there are enough points to calculate mse
#         if len(roi_list)>3:
#             intensity_list = []
#             roi_avg = [sum(i[0] for i in roi_list)/len(roi_list), sum(i[1] for i in roi_list)/len(roi_list)]
#             roi_sub = np.subtract(roi_list, roi_avg)
#             roi_t = (sum(i*j for i, j in roi_list)*len(roi_list) - sum(i[0] for i in roi_list)*sum(i[1] for i in roi_list))
#             roi_b = (sum(i[0]*i[0] for i in roi_list)*len(roi_list) - sum(i[0] for i in roi_list)**2)
#             if roi_t == 0 :
#                 roi_a = 100
#             elif roi_b ==0:
#                 roi_a = 0
#             else:
#                 roi_a = (-1)*roi_b/roi_t
#             # roi_a, roi_b = np.polyfit(roi_x, roi_y, 1)
#             i, j = x, y
#             pix = 0
#             add = 0
#             if roi_a<-1 or roi_a>=1:
#                 while i * j > 0 and i<iW and j<iH and mask[j, i] > 0:
#                     intensity_list.insert(0, img[j,i])
#                     pix += 1
#                     add+=1
#                     j += 1
#                     if roi_a < -10 or roi_a > 10:
#                         continue
#                     i=round(i+add/roi_a)
#                 jmax=j
#                 p1=(i,jmax)
#                 add=0
#                 i, j = x, y
#                 while i*j>0 and i<iW and j<iH and mask[j,i]>0:
#                     intensity_list.append(img[j,i])
#                     pix+=1
#                     add -= 1
#                     j-=1
#                     if roi_a < -10 or roi_a > 10:
#                         continue
#                     i=round(i+add/roi_a)
#                 jmin=j
#                 p2=(i,jmin)
#             else:
#                 while i * j > 0 and i<iW and j<iH and mask[j, i] > 0:
#                     intensity_list.insert(0, img[j,i])
#                     pix += 1
#                     i += 1
#                     add+=1
#                     if roi_a > -0.1 or roi_a < 0.1:
#                         continue
#                     j = round(j + add * roi_a)
#                 imax=i
#                 p1=(imax,j)
#                 add=0
#                 i, j = x, y
#                 while i*j>0 and i<iW and j<iH and mask[j,i]>0:
#                     intensity_list.append(img[j,i])
#                     pix+=1
#                     i-=1
#                     add-=1
#                     if roi_a > -0.1 or roi_a < 0.1:
#                         continue
#                     j = round(j - add * roi_a)
#                 imin=i
#                 p2=(imin,j)
#             plt.subplot(224), plt.plot(intensity_list)
#             plt.show()
#             plt.pause(0.1)
