import numpy as np
np.set_printoptions(threshold=np.nan)
import numpy.ma as ma
import numpy.linalg as la
import cv2
import cv2.cv as cv
import math
import matplotlib.pyplot as plt
import time

# Detects objects by comparing image with a reference background
# Compare only hue and saturation; ignore value
# because illumination invariant
# Object mask: 0 means foreground, 1 means background

def object_mask(img, bg, offset_v = 0, offset_h = 0):
    hue_threshold = int(0.1 * 180)
    sat_threshold = int(0.1 * 256)
    hue_threshold_sq = hue_threshold**2
    sat_threshold_sq = sat_threshold**2
    n_rows = img.shape[0]
    n_cols = img.shape[1]
    mask = np.ones([n_rows, n_cols])
    bg_hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    for i in range(offset_v - 1, n_rows - 1):
        for j in range(offset_h - 1, n_cols - 1):
            hue_diff = int(img_hsv[i, j, 0]) - int(bg_hsv[i, j, 0])
            hue_diff_sq = hue_diff**2
            if (hue_diff_sq > hue_threshold_sq):
                mask[i, j] = 0
                #print "hue diff:", i , ",", j
                continue
            sat_diff = int(img_hsv[i, j, 1]) - int(bg_hsv[i, j, 1])
            sat_diff_sq = sat_diff**2
            if (sat_diff_sq > sat_threshold_sq):
                mask[i, j] = 0
                #print "saturation diff:", i , ",", j
                continue
    return mask

# Mask background with given color (default: black)

def mask_background(img, mask, mask_color=(0, 0, 0)):
    result = np.array(img, copy=True)
    result[mask == 1] = mask_color
    return result

# Print the frame width, frame height, frames per second 
# and frame count of the input video using cap.get
    
def video_info(cap):
    fwidth = cap.get(cv.CV_CAP_PROP_FRAME_WIDTH)
    fheight = cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv.CV_CAP_PROP_FPS)
    fcount = cap.get(cv.CV_CAP_PROP_FRAME_COUNT)
    print "Frame width: ", fwidth, "\nFrame height: ", fheight, "\nFrames per second: ", fps, "\nFrame count: ", fcount

def main():

    bg = cv2.imread("background.jpg", cv2.CV_LOAD_IMAGE_COLOR)
    
    # Read the video

    cap = cv2.VideoCapture("football_right.mp4")
    n_frames = cap.get(cv.CV_CAP_PROP_FRAME_COUNT)

    

    _,f = cap.read()

    # Skip rows before starting object detection
    
    offset_v = 211
    start_time = time.time()
    obj_mask = object_mask(f, bg, offset_v)
    print "Object mask completed in", (time.time() - start_time), "s"

    
    start_time = time.time()
    fg = mask_background(f, obj_mask)
    print "Mask background completed in", (time.time() - start_time), "s"
    cv2.imshow("foreground", fg)
    cv2.waitKey(0)
    
    

    
    cv2.destroyAllWindows()
    cap.release()

main()
print "Done!"
