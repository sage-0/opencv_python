import cv2
import numpy as np

cap = cv2.VideoCapture("./images/30km制限.avi")  # カメラを起動する (引数はカメラのデバイス番号)
# cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

while True:
    ret, frame = cap.read()  # フレームをキャプチャ

    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # グレースケール変換
    edges = cv2.Canny(gray, 50, 200, 3)  # Cannyエッジ検出
    color_dst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)  # ハフ変換
    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=50, maxLineGap=13)
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 線を描画
    for linep in linesP:
        x1, y1, x2, y2 = linep[0]
        cv2.line(color_dst, (x1, y1), (x2, y2), (0, 0, 255), 3)

    cv2.imshow("HoughLines", frame)  # 結果を表示
    cv2.imshow("HoughLinesP", color_dst)

    key = cv2.waitKey(2)  # キー入力を待つ
    if key == 27:  # エスケープキーで終了
        break

cap.release()  # カメラを解放
cv2.destroyAllWindows()  # ウィンドウを閉じる