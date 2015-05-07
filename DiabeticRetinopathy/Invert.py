__author__ = 'Vamshi'
import cv2
import scipy.misc
import os

for subdir, dirs, files in os.walk('E:/UNM/CS 529 - Intro to Machine Learning/Assignment 4/Data/sample'):
    for f in files:
        path = os.path.join(subdir, f)
        im = cv2.imread(path)
        resized_img = scipy.misc.imresize(im, 0.25)
        image = (255-resized_img)
        cv2.imwrite('E:/UNM/CS 529 - Intro to Machine Learning/Assignment 4/Data/Resized/sample/' + f, image)
        cv2.waitKey(0)