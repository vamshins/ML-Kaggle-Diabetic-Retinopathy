__author__ = 'Vamshi'

import cv2
import matplotlib.pyplot as plt
import os

for subdir, dirs, files in os.walk('E:/UNM/CS 529 - Intro to Machine Learning/Assignment 4/Data/Resized/sample'):
    for f in files:
        path = os.path.join(subdir, f)
        im = cv2.imread(path)
        img = cv2.imread(path)

        # Create SURF object. You can specify params here or later.
        # Here I set Hessian Threshold to 400
        surf = cv2.SURF(400)

        # Find keypoints and descriptors directly
        kp, des = surf.detectAndCompute(img,None)

        print f + " - " + str(len(kp))

        #
        # img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
        #
        # plt.imshow(img2),plt.show()