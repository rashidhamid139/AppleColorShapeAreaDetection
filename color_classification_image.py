import cv2
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier
import os
import sys
from crop_image import cropND
from contour_area_calculation import calculate_area_of_contour
from shape_detection import detect_shape

# read the test image
try:
    source_image = cv2.imread(sys.argv[1])
except:
    # source_image = cv2.imread('test3.jpeg')
    testimg = cv2.imread('test6.png')
    source_image = cropND(testimg, testimg.shape[0], testimg.shape[1])
    contour_area = calculate_area_of_contour('test6.png')
    detected_shape = detect_shape('test6.png')

prediction = 'n.a.'

# checking whether the training data is ready
PATH = './training.data'

if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
    # print ('training data is ready, classifier is loading...')
    pass
else:
    # print ('training data is being created...')
    open('training.data', 'w')
    color_histogram_feature_extraction.training()
    # print ('training data is ready, classifier is loading...')

# get the prediction
color_histogram_feature_extraction.color_histogram_of_test_image(source_image)
prediction = knn_classifier.main('training.data', 'test.data')
print('Detected color is:', prediction)
print("Avg Area is: ", contour_area)
print("Detected shape is: ", detected_shape)
# cv2.putText(
#     source_image,
#     'Prediction: ' + prediction,
#     (15, 45),
#     cv2.FONT_HERSHEY_PLAIN,
#     3,
#     200,
#     )

# # Display the resulting frame
# cv2.imshow('color classifier', source_image)
# cv2.waitKey(0)		
