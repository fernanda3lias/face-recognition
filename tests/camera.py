import cv2
import imutils
import numpy as np

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    frame = imutils.resize(frame, 640)
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Grayscale Frame", grayscale_frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

capture.release()
cv2.destroyAllWindows()