__author__ = 'Vamshi'

import cv2
import os
import numpy as np
import csv
from sklearn.neighbors import KNeighborsClassifier
import scipy.misc
import shutil


imagespath_train = 'E:/UNM/CS 529 - Intro to Machine Learning/Assignment 4/Data/train/'
imagespath_test = 'E:/UNM/CS 529 - Intro to Machine Learning/Assignment 4/Data/Resized/sampleTest'
imagespath_dest = 'E:/UNM/CS 529 - Intro to Machine Learning/Assignment 4/Data/Resized/train/'
temp = 'E:/UNM/CS 529 - Intro to Machine Learning/Assignment 4/Data/Resized/temp/'


def copyfiles(src, dest):
    """
    This function is used to copy the files from "src" directory to "dest" directory
    Parameters:
    -----------
    src - Source directory containing files to be copied
    dest  - Destination directory where files have to be copied
    """
    src_files = os.listdir(src)
    for file_name in src_files:
        full_path = os.path.join(src, file_name)
        if os.path.isfile(full_path):
            shutil.copy(full_path, dest)


def resizeimages(imagespath1):
    """
    This function is used to Resize and Invert the images.
    Parameters:
    -----------
    imagespath1 - Path where all the images have to be resized and inverted
    """
    if os.path.exists(temp):
        shutil.rmtree(temp)
    os.mkdir(temp)
    for subdir, dirs, files in os.walk(imagespath1):
        for f in files:
            path = os.path.join(subdir, f)
            print "Resize - " + path
            im = cv2.imread(path)
            resized_img = scipy.misc.imresize(im, 0.4)  # Resize the image to 40% of the original image.
            image = (255-resized_img)   # Invert the resized image
            cv2.imwrite(temp + f, image)    # Write the images to temporary location
    copyfiles(temp, imagespath_dest)    # Copy the images from temporary location to destination path.


def take_certain_radius(imagespath_dest):
    """
    This function is used to crop the images from the center up to certain radius to remove the borders.
    Parameters:
    -----------
    imagespath_dest - path where files have to be resized
    """
    if os.path.exists(temp):
        shutil.rmtree(temp)
    os.mkdir(temp)
    for subdir, dirs, files in os.walk(imagespath_dest):
        for f in files:
            path = os.path.join(subdir, f)
            print "Radius - " + path
            im = cv2.imread(path)
            if "left" in f:         # flipping the left eye images
                im = cv2.flip(im, 1)
            height, width, depth = im.shape
            print height, width, depth
            thresh = 132
            imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(imgray, (5,5), 0)
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
                print x,y                   # Co-ordinates of center of the image
                print width/2.0,height/2.0
                print width/2-x,height/2-y
                crop_img = im[height/4:(height/1.3), width/4:(width/1.3)]  # Crop the image [y: y + h, x: x + w]
                cv2.imwrite(temp + f, crop_img)
            else:
                cv2.imwrite(temp + f, im)   # Write the files to the temporary location
            os.remove(path)                 # Delete the files after processing
    copyfiles(temp, imagespath_dest)    # Copy the files again from temporary location to destination path


def train(imagespath_dest):
    """
    This function extracts the features and the train dataset is in the format <Filename, Level, Feature Count>.
    “Feature count” is taken as input data (X_train) and “Level” is taken as classes (Y_train).
    This data is input to the “KNeighborsClassifier” (Number of neighbors = 5) for training.
    Parameters:
    -----------
    imagespath_dest - path where image files are fetched from
    Returns:
    --------
    neigh - Classifier
    """
    imageindex = 0
    num_lines = sum(1 for line in open('trainLabels.csv'))
    f = open('trainLabels.csv')
    header = 'y'
    trainlabels = {}    # train labels dictionary
    for line in iter(f):
        if header == 'y':   # Skipping headers
            header = 'n'
            continue
        image, level = line.split(',')
        trainlabels[image + ".jpeg"] = level.rstrip('\n')
        imageindex += 1

    imageindex = 0
    csvfile = "kps.csv"
    X_train = []
    Y_train = []
    with open(csvfile, "w") as output:  # Write the train data to csv file in the format <Filename, Level, Feature Count>
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

                # Find the length of keypoints and descriptors
                len_kp = str(len(kp))

                # print f + "," + str(trainlabels.get(f)) + "," + str(len_kp)
                writer.writerow([f, str(trainlabels.get(f)), str(len_kp)])
                X_train.append([int(len_kp)])
                Y_train.append(int(trainlabels.get(f)))
                imageindex += 1
    print X_train
    print Y_train
    neigh = KNeighborsClassifier(n_neighbors=5)  # implementing the k-nearest neighbors vote.
    neigh.fit(X_train, Y_train)     # Fit the model using X_train as training data and Y_train as target values
    return neigh


def test(imagespath_test, neigh):
    """
    This function classifies the test images using KNeighborsClassifier.
    Parameters:
    -----------
    imagespath_test - path where image files are fetched from
    neigh - KNeighborsClassifier
    """
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

            # Find the length of keypoints and descriptors
            len_kp = str(len(kp))

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
    neigh = train(imagespath_dest)
    test(imagespath_test, neigh)
    exit(0)
