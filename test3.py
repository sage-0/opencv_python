import cv2 as cv
import numpy as np

# 画像を読み込む
image = cv.imread('/Users/sage/worker/opencv_python/images/eaf30bf7-9a67-4f27-be48-9c60e096a274.png')

# 画像をグレースケールに変換
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# 輪郭を検出
contours, _ = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# 輪郭の中で最大のものを見つける
max_contour = max(contours, key=cv.contourArea)

# 最大輪郭を囲む正方形の座標を計算
x, y, w, h = cv.boundingRect(max_contour)

# 画像を正方形に切り抜く
cropped_image = image[y:y+h, x:x+w]

def alha(img):
        # グレースケールに変換する。
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # 2値化する。
        thresh, binary = cv.threshold(gray, 230, 255, cv.THRESH_BINARY_INV)

        # 輪郭を抽出する。
        contours, hierarchy = cv.findContours(
            binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        # マスクを作成する。
        mask = np.zeros_like(binary)

        # 輪郭内部 (透明化しない画素) を255で塗りつぶす。
        cv.drawContours(mask, contours, -1, color=255, thickness=-1)

        # RGBA に変換する。
        rgba = cv.cvtColor(img, cv.COLOR_RGB2RGBA)

        # マスクをアルファチャンネルに設定する。
        rgba[..., 3] = mask

        # 保存する。
        cv.imwrite(r"./result.png", rgba)

alha(cropped_image)