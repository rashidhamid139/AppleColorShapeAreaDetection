import numpy as np
import cv2
import operator    

def cropND(img, x, y):
    start = tuple(map(lambda a, da: int(a//2)-int(da//2), img.shape, (int(x/2), int(x/2))))
    end = tuple(map(operator.add, start, (int(x/2), int(y/2))))
    slices = tuple(map(slice, start, end))
    return img[slices]