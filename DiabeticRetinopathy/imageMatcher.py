__author__ = 'Vamshi'

import cv2

img = cv2.imread('10_left.jpeg', cv2.CV_LOAD_IMAGE_GRAYSCALE);
img2 = cv2.imread('10_left_c.jpeg', cv2.CV_LOAD_IMAGE_GRAYSCALE);

fd = cv2.FeatureDetector_create('ORB')
kpts = fd.detect(img)
kpts2 = fd.detect(img2)

# Now that we have the keypoints we must describe these points (x,y)
# and match them.
descriptor = cv2.DescriptorExtractor_create("BRIEF")
matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")


# descriptors (we must describe the points in some way)
k1, d1 = descriptor.compute(img, kpts)
k2, d2 = descriptor.compute(img2, kpts2)

# match the keypoints
matches = matcher.match(d1, d2)

# similarity
print '#matches:', len(matches)
