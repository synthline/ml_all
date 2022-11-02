import cv2 as cv

img = cv.imread('photos/park.jpg')

cv.imshow('Boston', img)

#  #greyscale
#  gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#  cv.imshow('Gray', gray)

# Blur
blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)


# Canny Edge Detection 
canny = cv.Canny (blur, 125,175)
cv.imshow('Canny Edges', canny)

# Dilate the images
dilated = cv.dilate(canny, (7,7), iterations=1)
cv.imshow('Dilated', dilated)

# Eroding (reverses dilation)
eroded = cv.erode(dilated, (3,3), iterations=1)
cv.imshow('Eroded', eroded)

cv.waitKey(0)
