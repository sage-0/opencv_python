import cv2 as cv



face_cascade_path = './haarcascade_frontalface_default.xml'
eye_cascade_path = './haarcascade_eye.xml'
aidle = './images/akb3.jpg'
soccer = './images/soccer2.jpg'

face_cascade = cv.CascadeClassifier(face_cascade_path)
eye_cascade = cv.CascadeClassifier(eye_cascade_path)

def one():
    cap = cv.VideoCapture("./testmovie3.avi")
    # "./testmovie3.avi"
    while True:
        key= cv.waitKey(1)
        if key != -1:
            break
        ret,src = cap.read()
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3,minSize = (20,20), maxSize = (250,250))
        #default scaleFactor=1.3, minNeighbors=5, minSize = (50,50), maxSize = (150,150)
        #usbcam ,scaleFactor=1.1,minNeighbors=5,minSize = (50,50), maxSize = (450,450)
        # img scaleFactor=1.1,minNeighbors=3,minSize = (20,20), maxSize = (250,250)

        ratio=0.05
        for x, y, w, h in faces:
            # cv.rectangle(src, (x, y), (x + w, y + h), (255,0,0),2)
            face = src[y: y + h, x: x + w]
            face_roi_gray = gray[y: y + int(h/2), x: x + w]
            eyes = eye_cascade.detectMultiScale(face_roi_gray, minSize=(20,20), maxSize=(30,30))
            #usbcam minSize=(50,50), maxSize=(70,70)
            # img minSize=(20,20), maxSize=(30,30)
            small = cv.resize(src[y: y + h, x: x + w], None, fx=ratio, fy=ratio, interpolation=cv.INTER_NEAREST)
            src[y: y + h, x: x + w] = cv.resize(small, (w, h), interpolation=cv.INTER_NEAREST)
            for (ex, ey, ew, eh) in eyes:
                # cv.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0,255,0),2)
                print("目の検出数: ",len(eyes))
        cv.imshow("顔",src)
        print("顔の検出数: ",len(faces))

    cap.release()

def two():
    src = cv.imread(soccer)

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.005,minNeighbors=15,minSize = (20,20), maxSize = (50,50))

    for x, y, w, h in faces:
            cv.rectangle(src, (x, y), (x + w, y + h), (255,0,0),2)
            face = src[y: y + h, x: x + w]
            face_roi_gray = gray[y: y + h, x: x + w]
            eyes = eye_cascade.detectMultiScale(face_roi_gray,scaleFactor=1.0003, minSize=(10,10), maxSize=(50,50))
            for (ex, ey, ew, eh) in eyes:
                if ew < 20:
                    cv.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0,255,0),2)
                    print("目の検出数: ",len(eyes))

    cv.imshow("output",src)
    print("顔の検出数: ",len(faces))
    cv.waitKey()


one()
# two()


cv.destroyAllWindows()

# def two():