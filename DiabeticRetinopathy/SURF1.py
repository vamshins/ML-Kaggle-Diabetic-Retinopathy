__author__ = 'Vamshi'

import cv2
import os
import numpy as np
import csv
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
import scipy.misc
import shutil


imagespath_train = 'E:/UNM/CS 529 - Intro to Machine Learning/Assignment 4/Data/train/'
imagespath_test = 'E:/UNM/CS 529 - Intro to Machine Learning/Assignment 4/Data/Resized/sampleTest'
imagespath_dest = 'E:/UNM/CS 529 - Intro to Machine Learning/Assignment 4/Data/Resized/train/'
temp = 'E:/UNM/CS 529 - Intro to Machine Learning/Assignment 4/Data/Resized/temp/'


def copyfiles(src, dest):
    src_files = os.listdir(src)
    for file_name in src_files:
        full_file_name = os.path.join(src, file_name)
        if (os.path.isfile(full_file_name)):
            shutil.copy(full_file_name, dest)


def resizeimages(imagespath1):
    if os.path.exists(temp):
        shutil.rmtree(temp)
    os.mkdir(temp)
    for subdir, dirs, files in os.walk(imagespath1):
        for f in files:
            path = os.path.join(subdir, f)
            print "Resize - " + path
            im = cv2.imread(path)
            resized_img = scipy.misc.imresize(im, 0.4)
            image = (255-resized_img)
            cv2.imwrite(temp + f, image)
    copyfiles(temp, imagespath_dest)


def take_certain_radius(imagespath_dest):
    if os.path.exists(temp):
        shutil.rmtree(temp)
    os.mkdir(temp)
    for subdir, dirs, files in os.walk(imagespath_dest):
        for f in files:
            path = os.path.join(subdir, f)
            print "Radius - " + path
            im = cv2.imread(path)
            im = (255-im)
            if "left" in f:         # flipping the left eye images
                im = cv2.flip(im, 1)
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
                    cv2.imwrite(temp + f, im)
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
                cv2.imwrite(temp + f, crop_img)
            else:
                cv2.imwrite(temp + f, im)
            os.remove(path)
    copyfiles(temp, imagespath_dest)


def invertimages(imagespath_dest):
    if os.path.exists(temp):
        shutil.rmtree(temp)
    os.mkdir(temp)
    for subdir, dirs, files in os.walk(imagespath_dest):
        for f in files:
            path = os.path.join(subdir, f)
            print "Invert - " + path
            im = cv2.imread(path)
            image = (255-im)
            cv2.imwrite(temp + f, image)
            os.remove(path)
    copyfiles(temp, imagespath_dest)


def train(imagespath_dest):
    no_of_images = 3008
    imageindex = 0
    num_lines = sum(1 for line in open('trainLabels.csv'))
    f = open('trainLabels.csv')
    header = 'y'
    # trainlabels = np.chararray((num_lines, 2))   # (3008L, 2L)
    trainlabels = {}
    for line in iter(f):
        if header == 'y':
            header = 'n'
            continue        # Skipping headers
        image, level = line.split(',')
        # print image + " - " + level
        # trainlabels.append([image, level])
        trainlabels[image + ".jpeg"] = level.rstrip('\n')
        imageindex += 1

    imageindex = 0
    kp_array = []
    csvfile = "kps.csv"
    X_train = []
    Y_train = []
    with open(csvfile, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for subdir, dirs, files in os.walk(imagespath_dest):
            for f in files:
                path = os.path.join(subdir, f)
                img = cv2.imread(path, 0)

                # Create SURF object. You can specify params here or later.
                # Here I set Hessian Threshold to 400
                surf = cv2.SURF(400)

                # Find keypoints and descriptors directly
                kp, des = surf.detectAndCompute(img, None)
                len_kp = str(len(kp))
                # kp_array.append((f, trainlabels.get(f), len_kp))
                # print f + "," + str(trainlabels.get(f)) + "," + str(len_kp)
                writer.writerow([f, str(trainlabels.get(f)), str(len_kp)])
                X_train.append([int(len_kp)])
                Y_train.append(int(trainlabels.get(f)))
                imageindex += 1
    print X_train
    print Y_train
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X_train, Y_train)
    return neigh


def test(imagespath_test, neigh):
    X_test = []
    Y_test = []

    for subdir, dirs, files in os.walk(imagespath_test):
        for f in files:
            path = os.path.join(subdir, f)
            img = cv2.imread(path, 0)

            # Create SURF object. You can specify params here or later.
            # Here I set Hessian Threshold to 400
            surf = cv2.SURF(400)

            # Find keypoints and descriptors directly
            kp, des = surf.detectAndCompute(img, None)
            len_kp = str(len(kp))
            # kp_array.append((f, trainlabels.get(f), len_kp))
            # print f + "," + str(trainlabels.get(f)) + "," + str(len_kp)
            X_test.append([int(len_kp)])

    # print neigh.predict([[100]])
    print neigh.predict(X_test)

if __name__ == '__main__':
    if os.path.exists(imagespath_dest):
        shutil.rmtree(imagespath_dest)
    os.mkdir(imagespath_dest)
    resizeimages(imagespath_train)
    take_certain_radius(imagespath_dest)
    # invertimages(imagespath_dest)
    neigh = train(imagespath_dest)
    test(imagespath_test, neigh)
    exit(0)
