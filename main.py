import cv2 as cv

cap = cv.VideoCapture(1)
if not cap.isOpened():
  print("Cannot open camera")
  exit(1)

cascade = cv.CascadeClassifier("/Users/sage/worker/opencv_python/raw.githubusercontent.com_opencv_opencv_master_data_haarcascades_haarcascade_frontalface_default.xml")

while True:
  # Capture frame-by-frame
  ret, frame = cap.read()
  # if frame is read correctly ret is True
  if not ret:
    print("Can't receive frame (stream end?). Exiting ...")
    break
  # Our operations on the frame come here
  gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
  # Display the resulting frame
  cv.imshow('frame', gray)
  canny = cv.Canny(frame, 10, 50)

  rects = cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5,minSize=(30,30))

  for (x, y, w, h) in rects:
    cv.rectangle(frame, (x,y), (x+h,y+h), (0, 0, 255), 2)

  # HSV変換
  img_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV_FULL)

  cv.imshow('human', frame)
  cv.imshow('hsv', img_hsv)


  cv.imshow('canny',canny)
  if cv.waitKey(1) == ord('q'):
    break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()