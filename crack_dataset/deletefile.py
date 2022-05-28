from create_dataset.create_annotations import *
from collections import OrderedDict
import cv2
import datetime
import json
import os

root_dir = '../crack_dataset/'
img_dir = os.path.join(root_dir, 'train/images/')
m_dir = os.path.join(root_dir, 'train/masks/')

testdir=os.listdir(m_dir)
dir=testdir.copy()
traindir=os.listdir(img_dir)
for x in dir:
    if not traindir.__contains__(x):
        os.remove(m_dir+x)

