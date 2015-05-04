__author__ = 'Vamshi'

import cv2
import matplotlib.pyplot as plt

img = cv2.imread('E:/UNM/CS 529 - Intro to Machine Learning/Assignment 4/Data/train/217_left.jpeg',0)

# Create SURF object. You can specify params here or later.
# Here I set Hessian Threshold to 400
surf = cv2.SURF(400)

# Find keypoints and descriptors directly
kp, des = surf.detectAndCompute(img,None)

print len(kp)


img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)

plt.imshow(img2),plt.show()

# # Check upright flag, if it False, set it to True
# print surf.upright
# # False
#
# surf.upright = True
#
# # Recompute the feature points and draw it
# kp = surf.detect(img,None)
# img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
#
# plt.imshow(img2),plt.show()