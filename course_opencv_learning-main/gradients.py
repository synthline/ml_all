import cv2 as cv
import numpy as np


img = cv.imread('photos/qDhl7.jpg')
cv.imshow('Cat', img)

# FROM BASIC:
 #greyscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Canny Edge Detection 
canny = cv.Canny (gray, 125,175)
cv.imshow('Canny Edges', canny)


# Latplacing 
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('Laplacian', lap)


# Sobel 
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobel7 = cv.Sobel(gray, cv.CV_64F, 0, 1)
combined_sobel = cv.bitwise_or(sobelx, sobel7)
cv.imshow('Sobelcom', combined_sobel)






cv.waitKey(0)