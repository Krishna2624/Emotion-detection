import cv2
from ultralytics import YOLO
import math
import cvzone

model = YOLO('D:/Downloads/HumanFaceEmotion/trainedYolov8.pt')
cap = cv2.VideoCapture("D:/Downloads/HumanFaceEmotion/vid3.mp4")
# cap = cv2.VideoCapture(0)
classNames = {'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'}
classNames = model.names
while True:
    success, img = cap.read()
    img = cv2.resize(img, [1280, 720], interpolation=cv2.INTER_AREA)
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])
            cvzone.cornerRect(img, (x1, y1, w, h), l=10, t=3, rt=1)
            cvzone.putTextRect(img, f'{classNames[cls]}{","} {conf}', pos=(max(0, x1), max(35, y1 - 10)), scale=1,
                               thickness=2, offset=4, colorT=(255, 0, 0), colorR=(0, 0, 0))

    cv2.imshow("Image", img)
    k = cv2.waitKey(1)
    if k == 27:  # close on ESC key
        break

cap.release()
cv2.destroyAllWindows()