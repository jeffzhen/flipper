import numpy as np
import cv2, time,os
import flipper.flipper as fp
import matplotlib.pyplot as plt
filepath = "/home/jeffz/flipper/data/test.avi"

cap = cv2.VideoCapture(filepath)
detector = fp.FlipDetector(True)
#while(cap.isOpened()):
timer = time.time()
for i in range(800):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    DBG = detector.capture(gray)
    
    #cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print float(time.time() - timer)
cap.release()
cv2.destroyAllWindows()
plt.plot(DBG)
plt.show()
#print detector.current_input
#print detector.DIFF_FILTER
#plt.plot(detector.buffer[0][0])
#plt.plot(detector.buffer[0][149])
#plt.show()
#plt.plot(detector.buffer[1][0])
#plt.plot(detector.buffer[1][149])
#plt.show()
