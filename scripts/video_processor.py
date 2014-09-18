import numpy as np
import cv2
import flipper.flipper as fp
import matplotlib.pyplot as plt
filepath = "/home/jeffz/flipper/data/test.avi"

cap = cv2.VideoCapture(filepath)
detector = fp.FlipDetector(True)
#while(cap.isOpened()):
for i in range(250):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    DBG = detector.capture(gray)
    
    #cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print detector.DIFF_FILTER
plt.plot(DBG)
plt.show()
