import glob
from create_dataset.create_annotations import *
from collections import OrderedDict
import cv2
import datetime
import json
import os
import fnmatch
from PIL import Image
import numpy as np
import re
import pycocotools.coco as coco

root_dir = '../crack_dataset/'
image_dir = os.path.join(root_dir, 'train/images/')
mask_dir = os.path.join(root_dir, 'train/masks/')


INFO = {
    "description": "Crack Dataset",
    "url": "",
    "version": "0.1.0",
    "year": 2022,
    "contributor": "hwkwak",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "",
        "url": ""
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'crack',
        'supercategory': 'crack',
    }
]
category_ids = {"crack": 1}
category_colors = {"(255, 255, 255)":1}



# cv2.namedWindow('img_color')

listdir_=os.listdir(image_dir)
listdir=listdir_.copy()
masklistdir=os.listdir(mask_dir)
for x in listdir_:
    if not masklistdir.__contains__(x):
        listdir.pop(listdir.index(x))
for x in masklistdir:
    if not listdir.__contains__(x):
        masklistdir.pop(masklistdir.index(x))
listdir_.clear()
images_ = get_list(listdir, ".jpg")
images_.sort()
images = []
real_name = []
for idx, _ in enumerate(images_):
    images.append(images_[idx].split('/')[-1])
    real_name.append(images_[idx].split('\\')[-1].split('.')[0])

print(len(real_name))

stay = 1
imgs=[]
annos=[]
coco_output = {
    "info": INFO,
    "licenses": LICENSES,
    "categories": CATEGORIES,
    "images": [{}],
    "annotations": [{}]
}
coco_format = coco_output
image_id = 0
annotation_id = 0
for idx, data in enumerate(images):
    stay = 1
    file_data = OrderedDict()
    directory=image_dir + '/' + data.split('\\')[-1]
    img_color = cv2.imread(directory)
    height, width, ch = img_color.shape

    maskdir= mask_dir + data.split('\\')[-1]
    img_mask = cv2.imread(maskdir)
    mask_img=Image.open(maskdir).convert('1').convert("RGB")
    w, h = mask_img.size
    image_annotation = create_image_annotation(listdir[idx], w, h, image_id)
    imgs.append(image_annotation)
    sub_masks = create_sub_masks(mask_img, w, h)
    if sub_masks is None: continue
    hascolor=False
    for color, sub_mask in sub_masks.items():
        if not category_colors.keys().__contains__(color):
            continue
        hascolor=True
        category_id = category_colors[color]
        polygons, segmentations = create_sub_mask_annotation(sub_mask)
        multi_poly = MultiPolygon(polygons)
        annotation = create_annotation_format(multi_poly, segmentations, image_id, category_id, annotation_id)
        annos.append(annotation)
        annotation_id+=1
    image_id+=1
    if not hascolor:
        print('{} || {} : no mask'.format(image_id, data))
        continue

    coco_format["categories"] = create_category_annotation(category_ids)
    coco_format["images"], coco_format["annotations"], annotation_cnt = imgs, annos, annotation_id

    cocodir="../crack_dataset/{}.json".format('new_masks')
    with open(cocodir, "w") as outfile:
        json.dump(coco_format, outfile)

    img_result = img_color.copy()
    coco_file=coco.COCO(cocodir)
    annIds = coco_file.getAnnIds(imgIds=[image_id-1], catIds=[1], iscrowd=None)
    anns = coco_file.loadAnns(annIds)

    for i, ann in enumerate(anns):
        segs = ann["segmentation"]
        segs = [np.array(seg, np.int32).reshape((1, -1, 2)) for seg in segs]
        for seg in segs: cv2.drawContours(img_result, seg, -1, (0, 255, 0), 0)

    while(True):
        if stay == 1:
            cv2.imshow('img_color', img_result)
            waitkey = cv2.waitKey(0)

        if waitkey & 0xFF == 97:
            break
        if waitkey & 0xFF == 0x31:
            cv2.imshow('img_color', img_result)
            waitkey = cv2.waitKey(0)
            stay=0
        if waitkey & 0xFF == 0x32:
            cv2.imshow('img_color', img_color)
            waitkey = cv2.waitKey(0)
            stay=0
        if waitkey & 0xFF == 0x33:
            cv2.imshow('img_color', img_color)
            waitkey = cv2.waitKey(0)
            stay=0

    print('{} || {} : done'.format(image_id, data))

