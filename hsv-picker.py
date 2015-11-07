import numpy as np
np.set_printoptions(threshold=np.nan)
import numpy.linalg as la
import cv2
import cv2.cv as cv
import math
import matplotlib.pyplot as plt
import time

def on_mouse(event, x, y, flags, hsv):
    if event == cv.CV_EVENT_LBUTTONUP:
        print "(", y, ",", x, "):", hsv[y, x]

def main():
    # Read the .mp4 video using OpenCV Python API cv2.VideoCapture

    #cap = cv2.VideoCapture("football_right.mp4")

   
    #_,f = cap.read()
    
    f = cv2.imread("background.jpg", cv2.CV_LOAD_IMAGE_COLOR)
    f_hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    cv2.namedWindow('frame1')
    cv.SetMouseCallback('frame1', on_mouse, f_hsv)
    cv2.imshow('frame1', f)
    cv2.waitKey(0)
    
    
    
    #cap.release()

main()
print "Done!"
