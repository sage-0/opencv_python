import numpy as np
import cv2

# モデルファイルのパス
prototxt_path = './AIscan/deploy.prototxt'
caffemodel_path = './AIscan/res10_300x300_ssd_iter_140000.caffemodel'

cap = cv2.VideoCapture(1)

# 顔検出用のモデルを読み込む
net = cv2.dnn.readNet(prototxt_path, caffemodel_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 顔検出
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300), (104, 117, 123)))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype(int)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            text = f'Face {i + 1}: {confidence * 100:.2f}%'
            cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # 画像を表示
    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# カメラをリリースし、ウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()