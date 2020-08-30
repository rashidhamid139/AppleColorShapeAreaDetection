import imutils
import cv2
import numpy as np


class ShapeDetector:
    def __init__(self):
        pass
    def detect(self, c):
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        
        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            shape = "unidentified" if ar >= 0.95 and ar <= 1.05 else "circle"
        elif len(approx) == 5:
            shape = "pentagon"
        else:
            shape = "circle"
        return shape
def detect_shape(image_path):
    image = cv2.imread(image_path)
    resized = imutils.resize(image, width=300)
    ratio = image.shape[0] / float(resized.shape[0])
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    sd = ShapeDetector()

    i = 0
    con_dict = {'circle': 0, 'pentagon': 0, 'triangle': 0, 'unidentified': 0}
    for c in cnts:
        if i > 5:
            break
        shape = sd.detect(c)
        con_dict[shape] = con_dict[shape] + 1
        i += 1

    max_votes = sorted(con_dict)
    return max_votes[0]