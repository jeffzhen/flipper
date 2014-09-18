import numpy as np
import cv2

filepath = "/home/jeffz/flipper/data/test.avi"

cap = cv2.VideoCapture(filepath)

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print gray.shape, gray.ndim
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
