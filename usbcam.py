import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(1)
fgpg = cv2.createBackgroundSubtractorMOG2()
fgpg2 = fgbg = cv2.createBackgroundSubtractorKNN()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
while True:
    key= cv2.waitKey(1)
    if key != -1:
        break

    ret,img = cap.read()
    fgmask = fgpg.apply(img)
    fgmask2 = cv2.morphologyEx(fgmask,cv2.MORPH_OPEN, kernel)

    edges = cv2.Canny(img, 100, 200)
    edges1 = cv2.Canny(img, 50, 0)
    edges2 = cv2.Canny(img, 550, 150)



    cv2.imshow('original', img)
    cv2.imshow('canny_original: 100, 200', edges)
    cv2.imshow('canny_1: 50, 0', edges1)
    cv2.imshow('canny_2: 550, 150', edges2)
    cv2.imshow('MOG', fgmask)
    cv2.imshow('MOG2',fgmask2)

cap.release()
cv2.destroyAllWindows()