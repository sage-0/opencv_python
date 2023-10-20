import cv2
import numpy as np

# 画像を読み込む
image = cv2.imread('./images/Screenshot_20231019-110239~2.png', cv2.IMREAD_COLOR)  # 画像のファイルパスを指定

if image is None:
    print("Error: Could not load the image.")
    exit()

# グレースケール変換
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Cannyエッジ検出
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# ハフ変換による直線検出（Pなし）
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100)

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
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 結果を表示
cv2.imshow("Hough Lines (No P)", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
