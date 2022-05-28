import cv2
import numpy as np
import PIL
from PIL import Image
import glob
import matplotlib.pyplot as plt
import math
import imgmod
from numpy import linalg
import time
from skimage.draw import line

class Crack:
    def __init__(self,img, angle_w=72, angle_h=60, gray=True, mask_eval=None, mask_bitand=None):
        self.max_width=0
        self.min_width=0
        self.avg_width=0
        self.image(img, gray)
        '''self.mask_eval=mask_eval
        if mask_bitand:
            self.mask_bitand=mask_bitand
        elif mask_eval.any():
            mask_eval=mask_eval.astype('uint8')
            self.mask_bitand=self.mask_bitand(self.thinning(mask_eval), self.thinned)'''

    def image(self, img, togray):
        """Image which contains crack"""
        if togray:
            self.img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.img=img
        ret,self.mask_otsu=self.otsu_thres(img)
        print(ret)
        self.thinned=self.thinning(self.mask_otsu)

    def bin2img(self, mask):
        th_img = self.img.copy()
        th_img[:, :, 0] = (mask * color[0]).astype('uint8')
        th_img[:, :, 1] = (mask * color[1]).astype('uint8')
        th_img[:, :, 2] = (mask * color[2]).astype('uint8')
        return th_img

    def mask_bitand(self, mask1, mask2):
        if len(mask1.shape)==3:
            mask1=mask1[:,:,0]
        if len(mask2.shape) ==3:
            mask1 = mask2[:, :, 0]
        bitand = cv2.bitwise_and(mask1, mask2)
        return bitand

    def calc_pix(self,angle, dist, pix,full_angle=None ):
        """calculate mm per pixel,
        dist is in cm"""

        return 20*dist*math.fabs(math.tan(angle/2))/pix

    def otsu_thres(self, img):
        """Use OTSU's Threshold Binarization -
        first do gaussian filtering then Otsu's"""
        image=np.copy(img)
        # Gaussian filtering
        #blur=cv2.GaussianBlur(image, (15,15),0)
        blur=cv2.medianBlur(image,3, 0)
        # Otsu!
        blur=cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        ret, img_thres =cv2.threshold(blur, 0,255,cv2.THRESH_OTSU)
        img_thres = cv2.bitwise_not(img_thres)

        return ret,img_thres

    def thres(self, img, th):
        """customizable thresholding"""
        imge=np.copy(img)
        # histogram
        #h=np.histogram(image, )

    def compare_mask(self, mask):
        """comparing detected mask with binarized image"""
        if self.mask_otsu:
            crack_mask=cv2.bitwise_and(mask, self.mask_otsu)
        else:
            print("No image to compare")
            crack_mask=None
        return crack_mask

    def width_calc(self, mask):
        if self.img:
            crack_mask=self.compare_mask(mask)
        else:
            print("No image to compare")

    def thinning(self, mask):
        """thinning mask"""
        kernel=cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        img_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        img_th=cv2.ximgproc.thinning(img_open, img_open, thinningType=cv2.ximgproc.THINNING_GUOHALL)
        return img_th

    def intensity(self, image, msk, kernel, half=None):
        """mod convolution method, does not convolutes image but outputs intensity of surrounding pixels,
        depending on kernel size and shape"""
        img = image.copy()
        mask = msk.copy()
        (iH, iW) = img.shape[:2]
        (mH, mW) = mask.shape[:2]
        if iH != mH and iW != mW:
            print('different image and mask size')
            return None
        (kH, kW) = kernel.shape[:2]
        b = (kW - 1) // 2
        p = np.sum(kernel)
        img_out = np.zeros((iH, iW), dtype="float32")
        # BorderTypes : BORDER_CONSTANT | BORDER_REPLICATE | BORDER_REFLECT | BORDER_WRAP | BORDER_TRANSPARENT | BORDER_ISOLATED
        img_pad = cv2.copyMakeBorder(img, b, b, b, b, borderType=cv2.BORDER_REPLICATE)
        for y in np.arange(b, iH + b):
            for x in np.arange(b, iW + b):
                if half:
                    if x % 2 == 0 or y % 2 == 0:
                        continue
                if mask[y - b, x - b] == 0:
                    continue
                roi = img_pad[y - b:y + b + 1, x - b:x + b + 1]
                k = ((roi * kernel).sum()) / p
                img_out[y - b, x - b] = k
        return img_out.astype("uint8")

    def intensity_linear(self, image, msk, half=None):
        """mod convolution method, does not convolutes image but outputs intensity of surrounding pixels,
        depending on kernel size and shape"""
        img = image.copy()
        mask = msk.copy()
        (iH, iW) = img.shape[:2]
        (mH, mW) = mask.shape[:2]
        if iH != mH and iW != mW:
            print('different image and mask size')
            return None
        img_out = np.zeros((iH, iW), dtype="float32")
        # don't need to pad, no kernel
        # but need to consider calculation area to range(3,W-4)
        for y in np.arange(3, iH -4):
            for x in np.arange(3, iW - 4):
                if half:
                    if x % 2 == 0 or y % 2 == 0:
                        continue
                if mask[y, x] == 0:
                    continue
                roi = mask[y - 3:y + 3 + 1, x - 3:x + 3 + 1]
                arr=[]
                x1 = []
                y1 = []
                for rx in range(7):
                    for ry in range(7):
                        if roi[rx,ry]==1:
                            x1.append(rx)
                            y1.append(ry)
                if len(x1)<3:
                    continue
                A = np.vstack([x1,np.one(len(x))]).T
                m, c = np.linalg.listsq(A,y1,rcond=None)[0]
                m=-1/m



        return img_out.astype("uint8")

    def get_r(self, mask):
        (mH, mW) = mask.shape[:2]
        tmp=np.zeros((mH,mW))


    def intensity_histo(self, n=1):
        if not self.thinned.any():
            print('no mask')
            return None
        arr = []
        for i in range(1, n):
            kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (i * 2 - 1, i * 2 - 1))
            # f.add_subplot(1,n,i)
            conv = self.intensity(self.img_gray, self.thinned, kernel_ellipse)
            # plt.imshow(conv)
            intens = conv[conv > 0]
            arr.append(intens[8])
            #arr[i-1].append(intens[0].astype('int32'))
        return arr


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


#print(math.tan(37*math.pi/180))
#print(math.tan(30*math.pi/180))
 # d= 10

# ------------height---------------------
# calc(2765,1266,2765,1314,350,5) #60.105
# calc(2765,1218,2765,1314,350,10) #60.105
# calc(2765,1170,2765,1314,350,15) #60.105

# ------------width----------------------
# calc(2719,1742, 2976,1742,390,28) #71.816
# calc(2803,1698, 2986,1698,390,20) #71.985
# calc(2803,1698, 2941,1698,390,15) #71.689
# calc(2858,1742, 2976,1742,390,13) #72.423
# calc(2839,1741, 2976,1742,390,15) #72.085
# calc(2849,1698, 2941,1698,390,10) #71.689
# calc(2711,1697, 2757,1697,390,5) #71.689
# calc(2192,1694, 2202,1694,390,1) #67.217x


def otsu_thres( img):
    """Use OTSU's Threshold Binarization -
    first do gaussian filtering then Otsu's"""
    image = np.copy(img)
    blur = cv2.medianBlur(image, 5, 0)
    blur = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    ret, img_thres = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
    img_thres = cv2.bitwise_not(img_thres)
    img_thres=cv2.erode(img_thres,np.ones((5,5),np.uint8))
    return ret, img_thres

def con(img_eval, thin, kern):
    image = img_eval.copy()
    mask = thin.copy()
    (iH, iW) = image.shape[:2]
    (mH, mW) = mask.shape[:2]
    if iH != mH and iW != mW:
        print('different image and mask size')
        return None
    (kH, kW) = kern.shape[:2]
    b = (kW - 1) // 2
    p = np.sum(kern)
    img_out = np.zeros((iH, iW), dtype="float32")
    # BorderTypes : BORDER_CONSTANT | BORDER_REPLICATE | BORDER_REFLECT | BORDER_WRAP | BORDER_TRANSPARENT | BORDER_ISOLATED
    img_pad = cv2.copyMakeBorder(image, b, b, b, b, borderType=cv2.BORDER_REPLICATE)
    for y in np.arange(b, iH + b):
        for x in np.arange(b, iW + b):
            if mask[y - b, x - b] == 0:
                continue
            roi = img_pad[y - b:y + b + 1, x - b:x + b + 1]
            k = np.bitwise_and(roi, kern).sum()
            img_out[y - b, x - b] = k/p
    return img_out.astype("float")
# i=0
# ii=1
# while ii>0.99:
#     #while ii>0.9:
#     i += 1
#     kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (i * 2 - 1, i * 2 - 1))
#     # f.add_subplot(1,n,i)
#     print(ii)
#     #conv = crack.intensity(img, bitand, kernel_ellipse)
#     conv=con(im, bitand,kernel_ellipse)
#     # plt.imshow(conv)
#     intens = conv[bitand > 0]
#     ii=intens.sum()/len(intens)
#     arr.append(ii)
    #n+=1

def convolute_pixel_kernel(img_eval, thin):
    image = img_eval.copy()
    skeleton = thin.copy()
    (iH, iW) = image.shape[:2]
    (mH, mW) = skeleton.shape[:2]
    if iH != mH and iW != mW:
        print('different image and mask size')
        return None
    arr_out = np.zeros((iH, iW))
    # BorderTypes : BORDER_CONSTANT | BORDER_REPLICATE | BORDER_REFLECT | BORDER_WRAP | BORDER_TRANSPARENT | BORDER_ISOLATED
    for y in np.arange(0, iH):
        for x in np.arange(0, iW):
            if skeleton[y, x] == 0:
                continue
            ii, i = 1.0, 1
            while ii>0.99:
                kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (i * 2 - 1, i * 2 - 1))
                (kH, kW) = kern.shape[:2]
                b = (kW - 1) // 2
                p = np.sum(kern)
                img_pad = cv2.copyMakeBorder(image, b, b, b, b, borderType=cv2.BORDER_REPLICATE)
                roi = img_pad[y:y + 2*b + 1, x:x + 2*b + 1]
                k = np.bitwise_and(roi, kern).sum()
                ii = k / p
                i += 1
            arr_out[x, y]=i
            # draw ellipse on original image
            img_out = cv2.ellipse(imge, (x,y), (i,i), 0, 0, 360, (255,0,0))
    return arr_out, img_out
#arr=crack.intensity_histo(20)
#a=arr.copy()
#a.sort()
#min=a[0]
#arr=arr-min
#plt.plot(np.diff(arr))



'''conv3=con(img, img_th, kern)
f.add_subplot(1,2,1)
plt.imshow(conv3)

kern=kernel_9
conv9=con(img, img_th, kern)
f.add_subplot(1,2,2)
plt.imshow(conv9)
plt.show()'''

#cv2.imwrite('data/thinning/thinmask.png', comb)
'''gray=cv2.cvtColor(crack.img, cv2.COLOR_BGR2GRAY)
hist, _=np.histogram(gray.ravel(),range(256))
ax[2].plot(range(255), hist)
print('4')
dh=np.diff(hist)
ax[3].plot(range(254), dh)
print('5')
print(dh[dh==0])
ddh=np.diff(dh)
ax[4].plot(range(253), ddh)'''
