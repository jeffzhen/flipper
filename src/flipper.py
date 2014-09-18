import cv2
import numpy as np
import time, os, sys

class FlipDetector:
    def __init__(self):
        self.TARGET_RESOLUTION = (160, 240) ###The target resolution of image processing. The actual process_resolution will be decided by decide_image_res() method.
        self.process_resolution = self.TARGET_RESOLUTION ###The resolution at which the images will be processed at. Input images are first compressed to this resolution.
        self.input_resolution = None ###The resolution at which the images are inputed. I assume this doesn't change.
        self.current_input = None
        self.previous_input = None
        self.buffer_frame = 2 ###number of intermidiate frames to store.
        self.buffer = None ###a buffer for storing intermidiate frames.
        self.oned_stream = None
        
    def decide_image_res(self):
        if self.input_resolution == None:
            raise RuntimeError('decide_image_res() called prematurely before self.input_resolution is set.')
        if len(self.input_resolution) != 2 or self.input_resolution[0] <= 0 or self.input_resolution[1] <= 0:
            raise TypeError('self.input_resolution has to be a pair of positive numbers.')
        else:
            self.process_resolution = tuple((np.array(self.input_resolution) / np.round(np.array(self.input_resolution).astype('float32')/np.array(self.TARGET_RESOLUTION))).astype('int32'))

    def fill_buffer(self, image):
        self.previous_input = self.current_input
        self.current_input = image
        

    def process_buffer(self):
        if self.oned_stream == None:
            
    
    def flip(self): #analyse oned_stream return 'L' for left, 'R' for right, and 0 otherwise
        return 0
        
    def capture(self, image):
        ###input check
        if type(image).__module__ != np.__name__:
            raise TypeError('Only 2D numpy arrays are supported. Input image is not a numpy array.')
        if image.ndim != 2:
            raise TypeError('Only 2D numpy arrays are supported. Input image is not a 2D array.')
        
        ###first frame check
        if self.input_resolution == None:
            self.input_resolution = image.shape
            self.decide_image_res()
            self.buffer = np.empty((buffer_frame, self.process_resolution[0], self.process_resolution[1]), dtype = 'float32')
        elif self.input_resolution != image.shape:
            raise TypeError('Image resolution changed from %s to %s. Abort.'%(self.input_resolution, image.shape))
        
        ###fille buffer and process
        self.fill_buffer(image)
        self.process_buffer()
        return self.flip()
