"""
https://chrisalbon.com/code/machine_learning/preprocessing_images/isolate_colors/
https://stackoverflow.com/questions/16685707/why-is-the-range-of-hue-0-180-in-opencv
"""

import cv2
import numpy as np


img = cv2.imread("images/hsl.jpg")  # "images/hsl.jpg"
blur = 9

h_mean = 175
h_range = 15
s_ranges = (80, 255)
l_ranges = (80, 255)

# convert to HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# blur
img_hsv = cv2.GaussianBlur(img_hsv, (blur, blur), 0)

h, s, v = cv2.split(img_hsv)
cv2.imshow('h', h)
cv2.imshow('s', s)
cv2.imshow('v', v)


# create mask from hue range, including overflow
# important: opencv divides hue values (360°) by 2 -> range 0-180
h_lower = h_mean-h_range
h_upper = h_mean+h_range
limit_1 = np.array([h_lower,            s_ranges[0], l_ranges[0]])
limit_2 = np.array([h_upper,            s_ranges[1], l_ranges[1]])
mask = cv2.inRange(img_hsv, limit_1, limit_2)

if h_lower < 0:
    limit_1 = np.array([h_lower+179,   s_ranges[0], l_ranges[0]])
    limit_2 = np.array([179,           s_ranges[1], l_ranges[1]])
    mask_b = cv2.inRange(img_hsv, limit_1, limit_2)
    mask = cv2.bitwise_or(mask, mask_b)

elif h_upper > 179:
    limit_1 = np.array([0,             s_ranges[0], l_ranges[0]])
    limit_2 = np.array([h_upper-179,   s_ranges[1], l_ranges[1]])
    mask_b = cv2.inRange(img_hsv, limit_1, limit_2)
    mask = cv2.bitwise_or(mask, mask_b)


# Mask image
img_masked = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow("image", img)
cv2.imshow("mask", mask)
cv2.imshow("masked", img_masked)

cv2.waitKey(0)
cv2.destroyAllWindows()
