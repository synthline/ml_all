from turtle import width
import cv2 as cv

# img = cv.imread('photos/cat_large2.jpg')
# cv.imshow('Cat', img)


#works for videos, live and images
def rescaleFrame(frame, scale = 0.75):
    width = int(frame.shape[1] * scale) 
    height = int(frame.shape[0] * scale)
    dimmensions = (width, height)
    
    return cv.resize(frame, dimmensions, interpolation=cv.INTER_AREA)

#Only works for live video
def changeRes(width, height):
    capture.set(3, width)
    capture.set(4, height)

#Capture Videos
capture = cv.VideoCapture('videos/dog.mp4')

while True:
    isTrue, frame = capture.read()
    cv.imshow('Video', frame)
    
    frame_resized = rescaleFrame(frame)
    
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()   