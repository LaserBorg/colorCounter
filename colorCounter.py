"""
https://chrisalbon.com/code/machine_learning/preprocessing_images/isolate_colors/
https://stackoverflow.com/questions/16685707/why-is-the-range-of-hue-0-180-in-opencv
https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html
"""

import cv2
import numpy as np


img = cv2.imread("images/voting-red_green-cards.jpg")
blur = 1

h_mean = 55
h_range = 30
s_ranges = (50, 255)
l_ranges = (50, 255)

# convert to HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# blur
img_hsv = cv2.GaussianBlur(img_hsv, (blur, blur), 0)

# # preview HSV channels
# h, s, v = cv2.split(img_hsv)
# cv2.imshow('h', h)
# cv2.imshow('s', s)
# cv2.imshow('v', v)


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


# clean mask
d_shape = cv2.MORPH_ELLIPSE  # cv2.MORPH_RECT  cv2.MORPH_CROSS
d_size = 1
element = cv2.getStructuringElement(d_shape, (2 * d_size + 1, 2 * d_size + 1), (d_size, d_size))
mask = cv2.dilate(mask, element)
d_size = 2
element = cv2.getStructuringElement(d_shape, (2 * d_size + 1, 2 * d_size + 1), (d_size, d_size))
mask = cv2.erode(mask, element)


# find contours
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# loop over the contours
MIN_THRESH = 200
i = 1
for c in cnts[0]:
    if cv2.contourArea(c) > MIN_THRESH:
        # compute the center of the contour
        M = cv2.moments(c)

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # draw the contour and center of the shape on the image
        cv2.drawContours(img, [c], -1, (0, 255, 0), 1)
        cv2.putText(img, str(i), (cX-5, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        i += 1


# cv2.imshow("mask", mask)

# # masked image
# img_masked = cv2.bitwise_and(img, img, mask=mask)
# cv2.imshow("masked", img_masked)

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
