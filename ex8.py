import cv2

path = "./images/s_gomashio.png"
path2 = "images/s_gomashio2.jpg"

cap = cv2.VideoCapture(1)

gomashio = cv2.imread(path)
gomashio1 = cv2.imread(path2)

smoothing = cv2.blur(gomashio, (4, 4))
smoothing1 = cv2.blur(gomashio1, (8, 8))

cv2.imshow("gomashio", smoothing)
cv2.imshow("original", gomashio)
cv2.imshow("gomashio1", smoothing1)



while True:
    key=cv2.waitKey(1)
    if key != -1:
        break

    ret,img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret_h, threshhold = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    ret_h_o, threshhold_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    blur = cv2.blur(img, (7, 7))
    cv2.imshow("blur", blur)
    gaussianblur= cv2.GaussianBlur(img, (7, 7), 3)
    cv2.imshow("gaussianblur", gaussianblur)
    medianblur = cv2.medianBlur(img, 7)
    cv2.imshow("medianblur", medianblur)

    cv2.imshow("threshhold", threshhold)
    cv2.imshow("threshhold_otsu", threshhold_otsu)

    bilateral = cv2.bilateralFilter(img, 9, 75, 75)
    cv2.imshow("bilateralFilter", bilateral)

    cv2.imshow("Laplacian_CV_64F", cv2.Laplacian(img, cv2.CV_64F))
    cv2.imshow("Laplacian_CV_32F", cv2.Laplacian(img, cv2.CV_32F))
    cv2.imshow("Laplacian_CV_8U", cv2.Laplacian(img, cv2.CV_8U))
    cv2.imshow("Canny", cv2.Canny(img, 100, 200))
    cv2.imshow("Sobel_ksize=1_CV_32F", cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1))
    cv2.imshow("Sobel_ksize=1_CV_64F", cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1))
    cv2.imshow("Sobel_ksize=1_CV_8U", cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3))
    cv2.imshow("Sobel_ksize=15_CV_32F", cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=15))
    cv2.imshow("Sobel_ksize=15_CV_64F", cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=15))
    cv2.imshow("Sobel_ksize=15_CV_8U", cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=15))




cap.release()
cv2.destroyAllWindows()
cv2.waitKey()
