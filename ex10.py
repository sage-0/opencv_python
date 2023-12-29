import cv2 as cv

face_cascade_path = './haarcascade_frontalface_default.xml'
eye_cascade_path = './haarcascade_eye.xml'
arashi_path = './images/arashi1.jpg'
cat_path = './images/cat_ear2.png'
heart = './images/heart3.png'
eye_path = './images/eye.png'

face_cascade = cv.CascadeClassifier(face_cascade_path)
eye_cascade = cv.CascadeClassifier(eye_cascade_path)

def one():
    cap = cv.VideoCapture("./testmovie3.avi")
    # "./testmovie3.avi"
    while True:
        key = cv.waitKey(1)
        if key != -1:
            break
        ret, src = cap.read()
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20), maxSize=(250, 250))
        # default scaleFactor=1.3, minNeighbors=5, minSize=(50, 50), maxSize=(150, 150)
        # usbcam, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50), maxSize=(450, 450)
        # img scaleFactor=1.1, minNeighbors=3, minSize=(20, 20), maxSize=(250, 250)

        ratio = 0.05
        for x, y, w, h in faces:
            # cv.rectangle(src, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = src[y: y + h, x: x + w]
            face_roi_gray = gray[y: y + int(h / 2), x: x + w]
            eyes = eye_cascade.detectMultiScale(face_roi_gray, minSize=(20, 20), maxSize=(30, 30))
            # usbcam minSize=(50, 50), maxSize=(70, 70)
            # img minSize=(20, 20), maxSize=(30, 30)
            small = cv.resize(src[y: y + h, x: x + w], None, fx=ratio, fy=ratio, interpolation=cv.INTER_NEAREST)
            src[y: y + h, x: x + w] = cv.resize(small, (w, h), interpolation=cv.INTER_NEAREST)
            for (ex, ey, ew, eh) in eyes:
                # cv.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                print("目の検出数: ", len(eyes))
        cv.imshow("顔", src)
        print("顔の検出数: ", len(faces))

    cap.release()


def overlay_faces():
    frame = cv.imread(arashi_path)
    cat_ear = cv.imread(cat_path, cv.IMREAD_UNCHANGED)
    height, width, channels = cat_ear.shape
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    for x, y, w, h in faces:
        rate = w / width
        cat_ear_resized = cv.resize(cat_ear, (int(width * rate), int(height * rate)))
        face = frame[y: y + h, x: x + w]
        face_roi_gray = gray[y: y + h, x: x + w]
        eyes = eye_cascade.detectMultiScale(face_roi_gray)
        x1, y1, x2, y2 = x, y - int(h * 0.3), x + cat_ear_resized.shape[1], y - int(h * 0.3) + cat_ear_resized.shape[0]
        frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] * (1 - cat_ear_resized[:, :, 3:] / 255) + cat_ear_resized[:, :, :3] * (cat_ear_resized[:, :, 3:] / 255)
        for (ex, ey, ew, eh) in eyes:
            if ew < 20:
                cv.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                print("目の検出数: ", len(eyes))

    cv.imshow("output", frame)
    print("顔の検出数: ", len(faces))
    cv.waitKey()


def png_overlay():
    from pngoverlay import PNGOverlay
    frame = cv.imread(arashi_path)
    cat_ear = cv.imread(cat_path, cv.IMREAD_UNCHANGED)
    eye = cv.imread(eye_path, cv.IMREAD_UNCHANGED)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    height, width, channels = cat_ear.shape
    for x, y, w, h in faces:
        rate = w / width

        cat_ear_overlay = PNGOverlay(cat_path)
        cat_ear_overlay.resize(rate)
        cat_ear_overlay.show(frame, int(x+w/2), int(y-h*0.05))

        heart_overlay = PNGOverlay(heart)
        heart_overlay.resize(rate)
        heart_overlay.show(frame, int(x+w), int(y+h))
        face = frame[y: y + h, x: x + w]
        face_roi_gray = gray[y: y + h, x: x + w]
        eyes = eye_cascade.detectMultiScale(face_roi_gray)
        for (ex, ey, ew, eh) in eyes:
            eye_resized = cv.resize(eye, (ew, eh))

            # 一時ファイルに保存
            cv.imwrite('./tmp.png', eye_resized)

            # 保存したファイルを読み込んでOverlay
            eye_overlay = PNGOverlay('./tmp.png')
            for (ex, ey, ew, eh) in eyes:
                cv.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0,255,0), 2)
                eye_overlay.show(face, ex, ey)
            print("目の検出数: ", len(eyes))

    cv.imshow("PNGOverlay", frame)

def apply_task5():
    import os
    from pngoverlay import PNGOverlay

    cat_path = '.images/cat_ear.png'
    heart_path = '.images/heart.png'
    eye_path = '.images/eye.png'
    cap = cv.VideoCapture(1)

    while True:
        ret, frame = cap.read()

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray)

        for x, y, w, h in faces:
            face = frame[y:y+h, x:x+w]
            face_gray = gray[y:y+h, x:x+w]

            cat_ear = PNGOverlay(cat_path)
            cat_ear.resize(w / cat_ear.width)
            cat_ear.show(frame, int(x+w/2), int(y-h*0.05))

            heart = PNGOverlay(heart_path)
            heart.resize(w / heart.width)
            heart.show(frame, int(x+w), int(y+h))

            cv.rectangle(face, (x, y), (x+w, y+h), (255, 0, 0), 2)

            eyes = eye_cascade.detectMultiScale(face_gray)

            for ex, ey, ew, eh in eyes:
                eye_roi = cv.resize(eye, (ew, eh))
                cv.imwrite('./tmp.png', eye_roi)
                eye_overlay = PNGOverlay('./tmp.png')
                cv.rectangle(face, (ex, ey),
                              (ex+ew, ey+eh), (0, 255, 0), 2)

            eye_overlay.show(face, ex, ey)
            os.remove('./tmp.png')

        cv.imshow('frame', frame)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()


cv.destroyAllWindows()

if __name__ == '__main__':
    #one()
    # overlay_faces()
    png_overlay()
    # apply_task5()
    cv.waitKey()
    cv.destroyAllWindows()
