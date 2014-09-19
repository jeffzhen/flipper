import cv2
import numpy as np
import time, os, sys
import scipy.signal as ss
import matplotlib.pyplot as plt

class FlipDetector:
#######This calss handles the algorithm part. It takes in one frame at a time through capture(), and stores differential images in its buffer. 
#######Capture outputs a 'L' or 'M' or 'R' indicating movement direction.
#######To get debugging information, call get_detection_data() to retrieve the 1D time-filtered data
    def __init__(self, DBG = False):
        self.TARGET_RESOLUTION = (160, 240) ###The target resolution of image processing. The actual process_resolution will be decided by decide_image_res() method.
        self.process_resolution = self.TARGET_RESOLUTION ###The resolution at which the images will be processed at. Input images are first compressed to this resolution.
        self.input_resolution = None ###The resolution at which the images are inputed. I assume this doesn't change.
        self.current_input = None ###store an unchanged image exactly as it was inputed
        self.previous_input = None ###store an unchanged image exactly as it was inputed
        self.BUFFER_LEN = 2 ###number of intermidiate frames to store.
        self.buffer_ptr = 0 ###a pointer that loops through buffer frames. Outside fill_buffer(), this pointer always points at the frame that is one frame ahead of the latest frame (in another word the oldest frame)
        self.buffer = None ###a buffer for storing intermidiate frames.
        self.oned_stream = np.zeros(600, dtype='float32') ###we extract one flip_value out of each differential image and put it into this 1D stream. The final flip decision will be based on this 1D array. 600 works for roughly 20 seconds which should be enough for any future design
        self.image_count = 0 ###number of inputs handled
        self.FILTER_H_WIDTH = 0.01 ###half width of filter in terms of fraction of image width
        self.DIFF_FILTER = [[1]] ###place holder
        self.TIME_FILTER = np.ones(10)/10
        self.THRESHOLD = 100000 ###time-filtered value in oned_stream larger than this will be marked as detection
        self.BLIND_PERIOD = 60 ###number of frames after a detection that no detection will be marked
        self.last_detection = - self.BLIND_PERIOD - 1 ###the image_count at which last detection was made
        self.DBG = DBG
        
    def decide_image_res(self, input_shape):###decide what resolution to use for all the computation and create filters and buffers accordingly
        self.input_resolution = input_shape
        if len(self.input_resolution) != 2 or self.input_resolution[0] <= 0 or self.input_resolution[1] <= 0:
            raise TypeError('self.input_resolution has to be a pair of positive numbers.')
        else:
            self.process_resolution = tuple((np.array(self.input_resolution) / np.round(np.array(self.input_resolution).astype('float32')/np.array(self.TARGET_RESOLUTION))).astype('int32'))
            
            ###create filter to be convolved on differential images
            self.DIFF_FILTER = np.zeros((1, 2 * int(self.FILTER_H_WIDTH * self.process_resolution[1]) + 1), dtype='float32')
            self.DIFF_FILTER[0,0] = 1;
            self.DIFF_FILTER[0,-1] = -1;
            
            ###create buffer
            self.buffer = np.empty((self.BUFFER_LEN, self.process_resolution[0], self.process_resolution[1]), dtype = 'float32')
            
            
    def fill_buffer(self, image):###compute the differential image and put it into buffer.
        self.previous_input = self.current_input
        self.current_input = image ###if memory is of concern this is not necessary
        self.image_count = self.image_count + 1
        
        if type(self.previous_input).__module__  == np.__name__:
            if self.DBG:
                print "F%i: filling buffer frame %i"%(self.image_count, self.buffer_ptr)
            dsize = (self.process_resolution[1],self.process_resolution[0]) ###dsize has opposite convention as array dimensions
            self.buffer[self.buffer_ptr] = cv2.resize(self.current_input, dsize = dsize).astype('float32') - cv2.resize(self.previous_input, dsize = dsize).astype('float32')
            self.buffer_ptr = (self.buffer_ptr + 1) % self.BUFFER_LEN
        

    def process_buffer(self):###compute one single number indicating the leftness or rightness using the differential images stored in buffer, and append it to the 1D list oned_stream
        if self.image_count < 3:###need two differential frames meaning 3 input frames
            return
        
        bufferptr1 = (self.buffer_ptr - 1) % self.BUFFER_LEN
        bufferptr2 = (self.buffer_ptr - 2) % self.BUFFER_LEN
        
        flip_value = np.sum(ss.convolve(self.buffer[bufferptr2], self.DIFF_FILTER, mode='same') * self.buffer[bufferptr1])
        self.oned_stream[:-1] = self.oned_stream[1:]
        self.oned_stream[-1] = flip_value
    
    def flip(self): #analyse oned_stream return 'L' for left, 'R' for right, and 'M' otherwise
        tfiltered = ss.convolve(self.oned_stream[-len(self.TIME_FILTER):], self.TIME_FILTER, mode='valid')[0]
        if self.image_count > self.last_detection + self.BLIND_PERIOD:
            if tfiltered > self.THRESHOLD:
                self.last_detection = self.image_count
                return 'L'
            elif tfiltered < -self.THRESHOLD:
                self.last_detection = self.image_count
                return 'R'
        return 'M'
            
        
    def get_detection_data(self):
        return ss.convolve(self.oned_stream, self.TIME_FILTER, mode='valid')[-self.image_count:]
        
    def capture(self, image):
        ###input check
        if type(image).__module__ != np.__name__:
            raise TypeError('Only 2D numpy arrays are supported. Input image is not a numpy array.')
        if image.ndim != 2:
            raise TypeError('Only 2D numpy arrays are supported. Input image is not a 2D array.')
        
        ###first frame check
        if self.input_resolution == None:
            self.decide_image_res(image.shape)###decide what resolution to use for all the computation and create filters and buffers accordingly
        elif self.input_resolution != image.shape:
            raise TypeError('Image resolution changed from %s to %s. Abort.'%(self.input_resolution, image.shape))
        
        ###fille buffer and process
        self.fill_buffer(image)
        self.process_buffer()
        return self.flip()


class CamProcessor:
#######This class automatically connects to an available camera and feeds its images into a FlipDetector. 
#######It draws the 1D signal curve that is used by FlipDetector for deciding movements.
    def __init__(self):
        self.cap = None###video capture object that will be started in start_cap
        self.detector = FlipDetector()###FlipDetector handles all the algorithms
        self.frame = None###current fram from cam that is being processed
        self.DISPLAY_COUNT = 15###Number of frames to display the symbol when a detection is made. Without this, a detection will only flash up for 1 frame and not noticeable. This is purely cosmetic and does not affect the intrinsic blind period in the FlipDetector's algorithm
        self.display_count = 0###a counter that decreases from DISPLAY_COUNT to 0 after a detection. DETECTION_SYMBOL is displayed when this is positive.
        self.DETECTION_SYMBOL = {'L': np.array([[-2,0],[0,2],[0,1],[2,1],[2,-1],[0,-1],[0,-2]])/10., 'R':np.array([[2,0],[0,2],[0,1],[-2,1],[-2,-1],[0,-1],[0,-2]])/10.}###shape of the symbol to indicate left or right
        self.DETECTION_SYMBOL_OFFSET = {'L':[.2,.2], 'R':[.8,.2]}###relative position in the image to diaplay is simbols. coordinates are (horizontal right, vertical down)
        self.SYMBOL_COLOR = [0,0,0]
        self.DISPLAY_CAM = True###Whether to display camera image or simply display symbols on blank screen
        self.detection = None###the latest detection of 'L' or 'R'
        self.DRAW_INTERVAL = 5###number of frame interval at which the debugging 1d data curve is drawn.
        self.PLOT_SIZE = 1###an arbitrary relative linear scale. change this if you want to tweak the size of the 1d dta curve's size
    
    def start_cap(self):###start video capture object
        self.cap = cv2.VideoCapture(-1)
        
       
    def run(self):
        ###start video object
        self.start_cap()
        
        #initialize plotting stuff
        plt.ion()##enable dynamic plotting
        self.fig, self.ax = plt.subplots()
        self.plot, = self.ax.plot([], [], lw=2)
        plt.ylim([-2000000, 2000000])
        
        #start while loop
        i = 0
        while(self.cap.isOpened()):
            try:
                i = i + 1
                
                ####capture
                ret, self.frame = self.cap.read()
                ####convert image into grayscale
                gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                ####feed into FlipDetector
                signal = self.detector.capture(gray)
                ###draw data curve for debugging
                if i%self.DRAW_INTERVAL == 0:
                    data = self.detector.get_detection_data()
                    self.plot.set_data(np.arange(len(data)) + self.detector.image_count - len(data), data)
                    plt.xlim([self.detector.image_count-len(data), self.detector.image_count])
                    plt.title("FRAME %i: %s"%(self.detector.image_count,signal))
                    self.fig.set_size_inches(gray.shape[1]/100.*self.PLOT_SIZE, gray.shape[0]/100.*self.PLOT_SIZE)
                    plt.draw()
                ###display symbols and camera images
                self.display(signal)
                
                cv2.waitKey(1)
            except KeyboardInterrupt:
                self.stop()
            
    def display(self, signal):###display symbols and camera images
        if signal != 'M':
            self.display_count = self.DISPLAY_COUNT
            self.detection = signal
        else:
            self.display_count = self.display_count - 1
            
        if not self.DISPLAY_CAM:###hacky trick to get white background
            self.frame = self.frame * 0.5
        
        if self.display_count > 0:###a counter that decreases from DISPLAY_COUNT to 0 after a detection. DETECTION_SYMBOL is displayed when this is positive.
            cv2.fillConvexPoly(self.frame, (self.DETECTION_SYMBOL[self.detection]*self.frame.shape[0] + np.array([self.DETECTION_SYMBOL_OFFSET[self.detection] for i in range(len(self.DETECTION_SYMBOL[self.detection]))])*self.frame.shape[1]).astype(int), self.SYMBOL_COLOR)
            
        ###final display. We flip the cameraso it's mirror image and intuitive
        cv2.imshow('frame', cv2.resize(cv2.flip(self.frame, 1), dsize=(1000, 750)))
            
    def stop(self):
        self.cap.release()
        cv2.destroyAllWindows() 


class VideoProcessor(CamProcessor):
#######This class takes a video file on hard drive and feeds its images into a FlipDetector. 
#######It draws the 1D signal curve that is used by FlipDetector for deciding movements.
    def __init__(self, file_name):
        CamProcessor.__init__(self)
        self.FILE_NAME = file_name
        
        
    def start_cap(self):###start video capture object
        self.cap = cv2.VideoCapture(self.FILE_NAME)
        
        
    def display(self, signal):
        if signal != 'M':
            self.display_count = self.DISPLAY_COUNT
            self.detection = signal
        else:
            self.display_count = self.display_count - 1
        
        if self.display_count > 0:###a counter that decreases from DISPLAY_COUNT to 0 after a detection. DETECTION_SYMBOL is displayed when this is positive.
            cv2.fillConvexPoly(self.frame, (self.DETECTION_SYMBOL[self.detection]*self.frame.shape[0] + np.array([self.DETECTION_SYMBOL_OFFSET[self.detection] for i in range(len(self.DETECTION_SYMBOL[self.detection]))])*self.frame.shape[1]).astype(int), self.SYMBOL_COLOR)
            
        ###final display. We don't flip the camera
        cv2.imshow('frame', cv2.resize(cv2.flip(self.frame, 1), dsize=(1000, 750)))
