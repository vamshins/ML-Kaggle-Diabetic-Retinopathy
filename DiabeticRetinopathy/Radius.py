__author__ = 'Vamshi'
import numpy as np
import cv2
import scipy.misc
from PIL import Image
from scipy import misc
import os

def copy(f, im):
    cv2.imwrite('E:/UNM/CS 529 - Intro to Machine Learning/Assignment 4/Data/sample/' + f, im)

if __name__ == '__main__':
    for subdir, dirs, files in os.walk('E:/UNM/CS 529 - Intro to Machine Learning/Assignment 4/Data/sample'):
        for f in files:
            path = os.path.join(subdir, f)
            im = cv2.imread(path)
            im = (255-im)
            # im = scipy.misc.imresize(im, 0.4)
            height, width, depth = im.shape
            print height, width, depth
            thresh = 132
            imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(imgray,(5,5),0)
            edges = cv2.Canny(blur,thresh,thresh*2)
            contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                cnt = contours[0]
                cv2.drawContours(im,contours,-1,(0,255,0),-1)

                #centroid_x = M10/M00 and centroid_y = M01/M00
                M = cv2.moments(cnt)
                try:
                    x = int(M['m10']/M['m00'])
                    y = int(M['m01']/M['m00'])
                except:
                    # copy(f, im)
                    print "exception"
                print x,y
                print width/2.0,height/2.0
                print width/2-x,height/2-y


                # cv2.circle(im,(x,y),1,(0,0,255),2)
                # cv2.putText(im,"center of Sun contour", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
                # cv2.circle(im,(width/2,height/2),1,(255,0,0),2)
                # cv2.putText(im,"center of image", (width/2,height/2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0))
                # cv2.imshow('contour',im)
                # resized_img = scipy.misc.imresize(im, 0.25)
                # cv2.imshow('', resized_img)
                # im = Image.open("10_left.jpeg")
                # crop_rectangle = (50, 50, 200, 200)
                # cropped_im = im.crop(crop_rectangle)
                # cropped_im.show()

                crop_img = im[height/4:(height/1.3), width/4:(width/1.3)] # Crop from x, y, w, h -> 100, 200, 300, 400
                # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
                cv2.imwrite('E:/UNM/CS 529 - Intro to Machine Learning/Assignment 4/Data/Resized/sample/' + f, crop_img)
            else:
                cv2.imwrite('E:/UNM/CS 529 - Intro to Machine Learning/Assignment 4/Data/Resized/sample/' + f, im)

