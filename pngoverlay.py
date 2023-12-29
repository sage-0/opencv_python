#--------------------------------------------------------------------------------
# OpenCVで透過PNGファイルの重ね合わせ
#
# 【使い方】：３ステップのみ
# (1)インポート
# import cv2
# from pngoverlay import PNGOverlay
#
# (2)インスタンス生成
# fish = PNGOverlay('img_fish.png')
#
# (3)表示メソッド
# fish.show(dst, x, y) #dstは表示させたい背景画像。x, yは表示したい中心座標
#
# 背景画像のヒント
# 例えば dst = cv2.imread('background.jpg') などで事前に読み込んでおく
#
# ---------- 以下は使わなくてもOK ----------
# （オプション）回転メソッド
# fish.rotate(45) #単位はdegree。showメソッド実行で反映
#
# （オプション）拡大・縮小メソッド
# fish.resize(0.5) #この例では0.5倍。showメソッド実行で反映
#--------------------------------------------------------------------------------

import cv2
import numpy as np

class PNGOverlay():
    def __init__(self, filename):
        # アルファチャンネル付き画像(BGRA)として読み込む
        self.src_init = cv2.imread(filename, -1)

        #必要最低限の透明色画像を周囲に付加する
        self.src_init = self._addTransparentImage(self.src_init)

        #画像の変形はデフォルトは不要
        self.flag_transformImage = False

        #画像の前処理を行う
        self._preProcessingImage(self.src_init)

        #初期値
        self.degree = 0
        self.size_value = 1

    def _addTransparentImage(self, src): #回転時にクロップしないように、予め画像の透明色領域を周囲に加える
        height, width, _ = src.shape # HWCの取得

        #回転対応で、対角線の長さを一辺とする透明色の正方形を作る
        diagonal = int(np.sqrt(width **2 + height ** 2))
        src_diagonal = np.zeros((diagonal, diagonal, 4), dtype=np.uint8)

        #透明色の正方形の中心にsrcを上書き
        p1 = int(diagonal/2 - width/2)
        p2 = p1 + width
        q1 = int(diagonal/2 - height/2)
        q2 = q1 + height
        src_diagonal[q1:q2,p1:p2,:] = src[:,:,:]

        return src_diagonal

    def _preProcessingImage(self, src_bgra): #BGRA画像をBGR画像(src)とA画像(mask)に分け、オーバーレイ時に必要な情報を保持
        self.mask = src_bgra[:,:,3]  # srcからAだけ抜き出し mask とする
        self.src = src_bgra[:,:,:3]  # srcからGBRだけ抜き出し src とする
        self.mask = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)  # Aを3チャンネル化
        self.mask = self.mask / 255.0  # 0.0-1.0に正規化
        self.height, self.width, _ = src_bgra.shape # HWCの取得
        self.flag_preProcessingImage = False #前処理フラグは一旦下げる

    def rotate(self, degree): #画像の回転パラメータ受付
        self.degree = degree
        self.flag_transformImage = True

    def resize(self, size_value): #画像のサイズパラメータ受付
        self.size_value = size_value
        self.flag_transformImage = True

    def _transformImage(self): #各メソッドでバラバラに行わず、一括でリサイズと回転をシリーズに行う必要がある
        #---------------------------------
        #resize
        #---------------------------------
        self.src_bgra = cv2.resize(self.src_init, dsize=None, fx=self.size_value, fy=self.size_value) #倍率で指定

        #サイズを変えたのでwidthとheightを出し直す
        self.height, self.width, _ = self.src_bgra.shape # HWCの取得

        #---------------------------------
        #rotate
        #---------------------------------
        #getRotationMatrix2D関数を使用
        center = (int(self.width/2), int(self.height/2))
        trans = cv2.getRotationMatrix2D(center, self.degree, 1)

        #アフィン変換
        self.src_bgra = cv2.warpAffine(self.src_bgra, trans, (self.width, self.height))

        #変形は終了したのでフラグはFalseにする
        self.flag_transformImage == False

        #オーバーレイの前に画像の前処理を行う
        self.flag_preProcessingImage = True

    def show(self, dst, x, y): #dst画像にsrcを重ね合わせ表示。中心座標指定
        #回転とサイズ変更はoverlayの直前で一括して行う必要がある
        if self.flag_transformImage == True:
            self._transformImage()

        #前処理が必要な場合は実行
        if self.flag_preProcessingImage == True:
            self._preProcessingImage(self.src_bgra)

        x1, y1 = x - int(self.width/2), y - int(self.height/2)
        x2, y2 = x1 + self.width, y1 + self.height #widthやheightを加える計算式にしないと１ずれてエラーになる場合があるので注意
        a1, b1 = 0, 0
        a2, b2 = self.width, self.height
        dst_height, dst_width, _ = dst.shape

        #x,y指定座標が dstから完全にはみ出ている場合は表示しない
        if x2 <= 0 or x1 >= dst_width or y2 <= 0 or y1 >= dst_height:
            return

        #dstのフレームからのはみ出しを補正
        x1, y1, x2, y2, a1, b1, a2, b2 = self._correctionOutofImage(dst, x1, y1, x2, y2, a1, b1, a2, b2)

        # Aの割合だけ src 画像を dst にブレンド
        dst[y1:y2, x1:x2] = self.src[b1:b2, a1:a2] * self.mask[b1:b2, a1:a2] + dst[y1:y2, x1:x2] * ( 1 - self.mask[b1:b2, a1:a2] )

    def _correctionOutofImage(self, dst, x1, y1, x2, y2, a1, b1, a2, b2): #x, y座標がフレーム外にある場合、x, y及びa, bを補正する
        dst_height, dst_width, _ = dst.shape
        if x1 < 0:
            a1 = -x1
            x1 = 0
        if x2 > dst_width:
            a2 = self.width - x2 + dst_width
            x2 = dst_width
        if y1 < 0:
            b1 = -y1
            y1 = 0
        if y2 > dst_height:
            b2 = self.height - y2 + dst_height
            y2 = dst_height

        return x1, y1, x2, y2, a1, b1, a2, b2

#テストコード
if __name__ == '__main__':
    dst = cv2.imread('image/mountain.jpg') #背景
    dst = cv2.resize(dst, dsize=None, fx=0.8, fy=0.8)

    #インスタンス生成
    happy = PNGOverlay('image/icon_happy.png')
    sad = PNGOverlay('image/icon_sad.png')
    test = PNGOverlay('image/belt.png')
    kurage = PNGOverlay('image/kurage_600px.png')

    #オーバーレイメソッド実行
    happy.show(dst, 200, 100)
    sad.show(dst, 500, 800)
    test.show(dst, 800, 500)
    kurage.resize(0.8)
    kurage.rotate(40)
    kurage.show(dst, 400, 500)
    cv2.imshow('dst',dst)
    cv2.waitKey(0)

    dst = cv2.imread('image/mountain.jpg') #背景
    dst = cv2.resize(dst, dsize=None, fx=0.8, fy=0.8)
    test.rotate(60)
    test.show(dst, 1200, 100)
    happy.rotate(-90)
    happy.resize(0.2)
    happy.show(dst, 800, 200)
    kurage.resize(0.3)
    kurage.rotate(-5)
    kurage.show(dst, 500, 500)

    cv2.imshow('dst',dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()