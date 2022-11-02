import cv2 as cv
import numpy as np

img = cv.imread('photos/cat.jpg')

cv.imshow('Cat', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)


# insert blur for cutting down number of contours
blur = cv.GaussianBlur(img, (5,5), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

# canny = cv.Canny(img, 125, 175)
# cv.imshow('Canny Edges', canny) 


# for returining external edges
# contours, hierarchies = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

# gives coordinates to the lines?
# contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)


# binarizes images?? 
ret, thresh = cv.threshold(gray, 125, 125, cv.THRESH_BINARY)
cv.imshow('Thresh', thresh)

contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
#prints number of countours found
print(f'{len(contours)} contour(s) found')

cv.waitKey(0)