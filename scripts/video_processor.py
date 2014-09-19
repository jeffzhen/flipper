import numpy as np
import cv2, time,os
import flipper.flipper as fp
import matplotlib.pyplot as plt
filepath = "/home/jeffz/flipper/data/test.avi"
plt.ion()
cap = cv2.VideoCapture(filepath)
detector = fp.FlipDetector()


#while(cap.isOpened()):
timer = time.time()
display_count = 0
DISPLAY_COUNT = 15
fig, ax = plt.subplots()
plot, = ax.plot([], [], lw=2)
plt.ylim([-500000,500000])
for i in range(800):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    signal = detector.capture(gray)
    print signal
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
        if detection == 'L':
            cv2.circle(frame, (0, gray.shape[0]/2), gray.shape[0]/5, [0,255,0], thickness = -1)
        elif detection == 'R':
            cv2.circle(frame, (gray.shape[1], gray.shape[0]/2), gray.shape[0]/5, [0,255,0], thickness = -1)

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
