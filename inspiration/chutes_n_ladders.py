"""
https://stackoverflow.com/questions/66446444/counting-the-number-of-shapes-in-an-image
"""

import cv2
import numpy as np

# load image
img = cv2.imread("../images/chutes_n_ladders.jpg")

# mask
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mask = cv2.inRange(gray, 240, 255)

# dilate and invert
kernel = np.ones((3,3), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=1)
mask = cv2.bitwise_not(mask)

# contours
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# remove very large and very small contours
filtered = []
low = 1000
high = 100000
for con in contours:
    area = cv2.contourArea(con)
    if low < area < high:
        filtered.append(con)

# draw centers of each
print("Shapes: " + str(len(filtered)))
for con in filtered:
    M = cv2.moments(con)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    cv2.circle(img, (cx, cy), 10, (0, 200, 100), -1)

# show
cv2.imshow("Image", img)
cv2.imshow("Mask", mask)
cv2.waitKey(0)
