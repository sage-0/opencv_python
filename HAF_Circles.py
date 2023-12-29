import cv2

cap = cv2.VideoCapture("./images/標識4つ.avi")
# cap = cv2.VideoCapture(1)
# img = cv2.imread('./images/Screenshot_20231019-110239~2.png', cv2.IMREAD_COLOR)

while True:
    ret, original = cap.read()
    ret, img = cap.read()  # 動画を画像として読み込む

    if img is None:
        break  # 画像がない場合は終了

    tmp_img = img.copy()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g_blur = cv2.GaussianBlur(gray_img, (9, 9), 2)
    canyedge = cv2.Canny

    circles = cv2.HoughCircles(g_blur, cv2.HOUGH_GRADIENT, dp=5, minDist=g_blur.shape[0] / 6, param1=200, param2=80, minRadius=1, maxRadius=30)

    if circles is not None:
        circles = circles[0].astype(int)
        for i in circles:
            center = (i[0], i[1])
            radius = i[2]
            # 円の中心を描画
            cv2.circle(img, center, 3, (0, 255, 0), -1)
            # 円の輪郭を描画
            cv2.circle(img, center, radius, (0, 0, 255), 3)

    cv2.imshow("Gaussian", g_blur)
    cv2.imshow("Circle", img)

    if cv2.waitKey(30) & 0xFF == 27:  # Escキーで終了
        break

cap.release()
cv2.destroyAllWindows()