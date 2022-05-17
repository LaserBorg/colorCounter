"""
https://pyimagesearch.com/2016/02/01/opencv-center-of-contour/
"""

import cv2


MIN_THRESH = 500

# load the image, convert it to grayscale, blur it slightly, and threshold it
image = cv2.imread("../images/shapes_and_colors.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

cv2.imshow("q", thresh)
cv2.waitKey(0)

# find contours in the thresholded image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# loop over the contours
for i, c in enumerate(cnts[0]):
    if cv2.contourArea(c) > MIN_THRESH:
        # compute the center of the contour
        M = cv2.moments(c)

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # draw the contour and center of the shape on the image
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
        cv2.putText(image, str(i), (cX - 10, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# show the image
cv2.imshow("Image", image)
cv2.waitKey(0)
