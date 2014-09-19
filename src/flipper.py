import cv2
import numpy as np
import time, os, sys
import scipy.signal as ss

class FlipDetector:
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
        self.THRESHOLD = 10000 ###time-filtered value in oned_stream larger than this will be marked as detection
        self.BLIND_PERIOD = 90 ###number of frames after a detection that no detection will be marked
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
