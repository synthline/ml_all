import cv2 as cv
import numpy as np

# img = cv.imread('photos/cat.jpg')
# cv.imshow('Cat', img)

#blank image
blank = np.zeros((500,500,3), dtype='uint8')
cv.imshow('Blank', blank)

#Paint the image in a certain colour
# blank[:] = 0,255,0
# cv.imshow('Green', blank)

#Paint an image within certain areas
# blank[200:300, 300:400] = 0,255,0
# cv.imshow('Green', blank)
 
# Draw a Line
cv.line(blank, (0,0), (blank.shape[1]//2,blank.shape[0]//2), (0,255,0), thickness=1)
cv.imshow('Line', blank)

#Draw a rectangle
# cv.rectangle(blank, (0,0), (250,250), (0,255,0), thickness=2)
# cv.imshow('Rectangle', blank)

#Colors the rectangle green
# cv.rectangle(blank, (0,0), (250,250), (0,255,0), thickness=-1)
# cv.imshow('Rectangle', blank)

#Draw a circle
# cv.circle(blank, (250,250), 40, (0,0,244), thickness=3)
# cv.imshow('Circle', blank)

#Write text
cv.putText(blank, 'Hello', (225,225), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0,225,0))
cv.imshow('Text', blank)

cv.waitKey(0)

