import numpy as np
import cv2

img = cv2.imread('./images/lena.jpeg')
cv2.imshow('test',img)
cv2.waitKey()
cv2.destroyAllWindows()