import cv2
from pngoverlay import PNGOverlay

img = cv2.imread('./images/orange.png')

item = PNGOverlay('./images/nekomimi1.png')

item.resize(1)
item.show(img, 250, 250)

cv2.imshow('image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()