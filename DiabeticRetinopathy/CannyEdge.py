__author__ = 'Vamshi'
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

for subdir, dirs, files in os.walk('E:/UNM/CS 529 - Intro to Machine Learning/Assignment 4/Data/processed/run-normal/train/'):
    for f in files:
        path = os.path.join(subdir, f)
        img = cv2.imread(path,0)
        edges = cv2.Canny(img,100,200)

        plt.subplot(121),plt.imshow(img,cmap = 'gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(edges,cmap = 'gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        print path
        plt.show()