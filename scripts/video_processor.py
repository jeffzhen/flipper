#!/usr/bin/env python
import numpy as np
import cv2, time, os, sys
import flipper.flipper as fp
import matplotlib.pyplot as plt

N_ARG = 1

if len(sys.argv) != N_ARG + 1:
	print "Input Error: excpecting exactly %i argument(s) for video file path."%(N_ARG)
	quit() 

show_camera_image = False


video = fp.VideoProcessor(sys.argv[1])
video.run()

