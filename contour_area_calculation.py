import imutils
import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_area_of_contour(path):
        
    img = cv2.imread(path, 0)
    kernel = np.ones((5,5),np.uint8)
    img = cv2.erode(img,kernel,iterations = 1)
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_area_array = sorted([cv2.contourArea(x) for x in contours])
    contour_area = sum(contour_area_array[0:-1])/len(contour_area_array[0:-1])
    return contour_area