import cv2 as cv

img = cv.imread('./media/images/testface1.jpg')

cv.imshow('image', img)

canny = cv.Canny(img, 125, 177)
cv.imshow("Canny", canny)

cv.waitKey(0)