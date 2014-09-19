import numpy as np
import cv2, time, os, sys
import flipper.flipper as fp
import matplotlib.pyplot as plt
filepath = "/home/jeffz/flipper/data/test.avi"
display_count = 0
DISPLAY_COUNT = 15
DETECTION_SYMBOL = {'L': np.array([[-2,0],[0,2],[0,1],[2,1],[2,-1],[0,-1],[0,-2]]), 'R':np.array([[2,0],[0,2],[0,1],[-2,1],[-2,-1],[0,-1],[0,-2]])}
DETECTION_SYMBOL_OFFSET = {'L':[.1,.1], 'R':[.9,.1]}

cap = cv2.VideoCapture(filepath)
detector = fp.FlipDetector()

plt.ion()
fig, ax = plt.subplots()
plot, = ax.plot([], [], lw=2)
plt.ylim([-500000,500000])

timer = time.time()
i = 0
while(cap.isOpened()):
#for i in range(800):
    i = i + 1
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    signal = detector.capture(gray)
    print "\bFRAME %i: %s"%(detector.image_count,signal)
    sys.stdout.flush()
    if i%5 == 0:
        data = detector.get_detection_data()
        plot.set_data(range(len(data)), data)
        plt.xlim([detector.image_count-len(data), detector.image_count])
        plt.draw()
    if signal != 'M':
        display_count = DISPLAY_COUNT
        detection = signal
    else:
        display_count = display_count - 1
    
    if display_count > 0:
        cv2.fillConvexPoly(frame, (DETECTION_SYMBOL[detection]*gray.shape[0]/20 + np.array([DETECTION_SYMBOL_OFFSET[detection] for i in range(len(DETECTION_SYMBOL[detection]))])*gray.shape[1]).astype(int), [0,255,0])

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print float(time.time() - timer)
cap.release()
cv2.destroyAllWindows()

#print detector.current_input
#print detector.DIFF_FILTER
#plt.plot(detector.buffer[0][0])
#plt.plot(detector.buffer[0][149])
#plt.show()
#plt.plot(detector.buffer[1][0])
#plt.plot(detector.buffer[1][149])
#plt.show()
