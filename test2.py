import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from copy import deepcopy

def main():
    path = "/Users/sage/worker/opencv_python/images/eaf30bf7-9a67-4f27-be48-9c60e096a274.png"     # 画像パス
    img_BGR = cv.imread(path)   # 画像読み込み
    img = cv.cvtColor(img_BGR, cv.COLOR_BGR2RGB)
    kernel = np.ones((2,2), np.uint8)
    dilation = cv.erode(img, kernel, iterations=1)

    # 背景色の指定
    R_mode = 255
    G_mode = 255
    B_mode = 255

    # # 対象画像と同じサイズの配列(チャンネル1，要素0)を用意し，背景でない箇所は255に変換
    # mask = np.zeros((dilation.shape[0], dilation.shape[1]))
    # mask[(dilation[:,:,0] != R_mode) & (dilation[:,:,1] != G_mode) &  (dilation[:,:,2] != B_mode)] = 255

    # # 輪郭の検出
    # contours, hierarchy = cv.findContours(mask.astype("uint8"), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # # 輪郭内部を黒で塗りつぶして削除
    # for contour in contours:
    #     cv.drawContours(mask, [contour], -1, (0, 0, 0), thickness=cv.FILLED)

    # # 検出した輪郭に対する処理
    # # img_with_line = deepcopy(img_BGR)
    img_with_line = deepcopy(dilation)
    # for i in range(len(contours)):
    #     if cv.contourArea(contours[i]) > (dilation.shape[0] * dilation.shape[1]) * 0.005:
    #         img_with_line = cv.drawContours(img_with_line, contours, i, (0,255,0), 2)

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

    alha(img_with_line)

if __name__ == "__main__":
    main()


