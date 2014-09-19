#!/usr/bin/env python
import numpy as np
import cv2, time, os, sys
import flipper.flipper as fp
import matplotlib.pyplot as plt


show_camera_image = False


cam = fp.CamProcessor()
cam.DISPLAY_CAM = show_camera_image
cam.run()

