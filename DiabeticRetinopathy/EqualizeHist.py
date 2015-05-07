__author__ = 'Vamshi'

import cv2
import numpy as np
import os
import scipy.misc

for subdir, dirs, files in os.walk('E:/UNM/CS 529 - Intro to Machine Learning/Assignment 4/Data/sample'):
    for f in files:
        path = os.path.join(subdir, f)
        img = cv2.imread(path, 0)
        resized_img = scipy.misc.imresize(img, 0.5)
        equ = cv2.equalizeHist(resized_img)
        # res = np.hstack((img, equ)) #stacking images side-by-side
        cv2.imwrite('E:/UNM/CS 529 - Intro to Machine Learning/Assignment 4/Data/EqHist/' + f, equ)
