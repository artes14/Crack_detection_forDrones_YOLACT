import pycocotools.coco as coco
import matplotlib as plt

def coco2mask(anns):
    mask = coco.annToMask(anns[0])
    for i in range(len(anns)):
        mask += coco.annToMask(anns[i])

    return mask