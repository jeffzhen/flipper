import numpy as np
import cv2, time, os, sys
import flipper.flipper as fp
import matplotlib.pyplot as plt



cap = cv2.VideoCapture(-1)
detector = fp.FlipDetector()


display_count = 0
DISPLAY_COUNT = 15
DETECTION_SYMBOL = {'R': np.array([[-2,0],[0,2],[0,1],[2,1],[2,-1],[0,-1],[0,-2]])/10., 'L':np.array([[2,0],[0,2],[0,1],[-2,1],[-2,-1],[0,-1],[0,-2]])/10.}
DETECTION_SYMBOL_OFFSET = {'R':[.2,.2], 'L':[.8,.2]}
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
    #print "\rFRAME %i: %s"%(detector.image_count,signal)
    sys.stdout.flush()
    if i%5 == 0:
        data = detector.get_detection_data()
        plot.set_data(np.arange(len(data)) + detector.image_count - len(data), data)
        plt.xlim([detector.image_count-len(data), detector.image_count])
        plt.title("\rFRAME %i: %s"%(detector.image_count,signal))
        fig.set_size_inches(6, 4)
        plt.draw()
    if signal != 'M':
        display_count = DISPLAY_COUNT
        detection = signal
    else:
        display_count = display_count - 1
    
    gray = gray * 0.1
    if display_count > 0:
        cv2.fillConvexPoly(gray, (DETECTION_SYMBOL[detection]*gray.shape[0] + np.array([DETECTION_SYMBOL_OFFSET[detection] for i in range(len(DETECTION_SYMBOL[detection]))])*gray.shape[1]).astype(int), [0,255,0])

    cv2.imshow('frame', cv2.resize(gray, dsize=(1000, 750)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print float(time.time() - timer)
cap.release()
cv2.destroyAllWindows()
