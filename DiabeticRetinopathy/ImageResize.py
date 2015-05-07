__author__ = 'Vamshi'

import cv2
import numpy as np
import os
import scipy.misc

for subdir, dirs, files in os.walk('E:/UNM/CS 529 - Intro to Machine Learning/Assignment 4/Data/sample'):
    for f in files:
        path = os.path.join(subdir, f)
        img = cv2.imread(path)
        resized_img = scipy.misc.imresize(img, 0.25)  # , interp='bilinear', mode='RGB'
        cv2.imwrite('E:/UNM/CS 529 - Intro to Machine Learning/Assignment 4/Data/Resized/sample/' + f, resized_img)
