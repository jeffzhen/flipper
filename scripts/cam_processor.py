#!/usr/bin/env python
import numpy as np
import cv2, time, os, sys
import flipper.flipper as fp
import matplotlib.pyplot as plt
import sys

show_camera_image = True
if len(sys.argv) > 1:
    cam_source = int(sys.argv[1])
    print "Using User Choice Camera #%i"%cam_source
else:
    cam_source = 0

cam = fp.CamProcessor(cam_source = cam_source)
cam.DISPLAY_CAM = show_camera_image
cam.run()

