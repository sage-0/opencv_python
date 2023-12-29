import cv2
from pngoverlay import PNGOverlay

img = cv2.imread('./images/cat.jpg')
# img = cv2.resize(img,(400, 300))
# cv2.putText(img, 'Stay Home',(250, 300), cv2.FONT_HERSHEY_SIMPLEX,2.0,(0,255,255),8)
# cv2.rectangle(img,(120,220),(270,260),(20,20,20),-1)
# cv2.circle(img,(180,420), 70, (0, 255, 255), -1)
item = PNGOverlay('./images/orange.png')
# item.show(img, 400, 280)
item.resize(0.2)
item.rotate(15)
item.show(img, 185, 140)

cv2.imshow('image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()