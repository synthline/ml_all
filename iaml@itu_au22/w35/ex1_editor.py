import cv2

# path = "/inputs/dipsy.jpg"

window_name = 'image'
             
img = cv2.imread("img112.jpg")

cv2.imshow('window_name', img)
cv2.waitKey() 