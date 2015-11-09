import numpy as np
np.set_printoptions(threshold=np.nan)
import numpy.ma as ma
import numpy.linalg as la
import cv2
import cv2.cv as cv
import math
import matplotlib.pyplot as plt
import time
import random

# Detects objects by comparing image with a reference background
# Compare only hue and saturation; ignore value
# because illumination invariant
# Object mask: 0 means foreground, 1 means background

def object_mask(img, bg, offset_v = 0, offset_h = 0):
    hue_threshold = int(0.02 * 180)
    hue_threshold_sq = hue_threshold**2
    n_rows = img.shape[0]
    n_cols = img.shape[1]
    mask = np.ones([n_rows, n_cols], dtype=np.int)
    bg_hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    obj_x = []
    obj_y = []
    
    for i in range(offset_v - 1, n_rows - 1):
        for j in range(offset_h - 1, n_cols - 1):
            hue = int(img_hsv[i, j, 0])
            bg_hue = int(bg_hsv[i, j, 0])

            # Retrieve foreground only if background
            # color of green field
            
            if not on_field_background(bg_hue):
                continue
            hue_diff = hue - bg_hue
            hue_diff_sq = hue_diff**2
            if (hue_diff_sq > hue_threshold_sq): 
                mask[i, j] = 0
                obj_x.append(i)
                obj_y.append(j)
                continue
    return mask, np.array(obj_x, dtype=np.int), np.array(obj_y, dtype=np.int)

# Hue of green color field

def on_field_background(hue):
    if hue in range(40, 50):
        return True
    return False

# Mask background with given color (default: black)

def mask_background(img, mask, mask_color=(0,0,0)):
    result = np.array(img, copy=True)
    result[mask == 1] = mask_color
    return result

# Features comprises:
# x coord (row index)
# y coord (column index)
# hue value
# saturation value

def feature_vector(img, obj_x, obj_y, x_max, y_max, increment=1):
    n_features = 3
    result = np.zeros([1, n_features])
    result_norm = np.zeros([1, n_features])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    n_points = obj_x.size
    for i in xrange(0, n_points, increment):

        x = obj_x[i]
        y = obj_y[i]
        hue = hsv[x, y, 0]
        #sat = hsv[x, y, 1]

        fv = np.array([x, y, hue])
        result = np.vstack((result, fv))

        # Normalize the values

        x = x / (x_max - 1)
        y = y / (y_max - 1)
        hue = hue / 179.
        #sat = sat / 255.

        fv = np.array([x, y, hue])
        result_norm = np.vstack((result_norm, fv))
        
    result_norm = np.delete(result_norm, 0, 0)
    result = np.delete(result, 0, 0)
    return result_norm, result

def show_images(original_frame, foreground):
    #cv2.imshow("original frame", original_frame)
    #cv2.imshow("foreground", foreground)
    cv2.imwrite("foreground-original.jpg", original_frame)
    cv2.imwrite("foreground.jpg", foreground)
    #cv2.waitKey(0)

def main():

    bg = cv2.imread("background-left.jpg", cv2.CV_LOAD_IMAGE_COLOR)
    
    # Read the video

    cap = cv2.VideoCapture("football_left.mp4")
    n_frames = cap.get(cv.CV_CAP_PROP_FRAME_COUNT)
    fwidth = cap.get(cv.CV_CAP_PROP_FRAME_WIDTH)
    fheight = cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv.CV_CAP_PROP_FPS)
    print "Frame - width:", fwidth, ", height:", fheight, ", FPS:", fps, ", # Frames:", n_frames

    
    for i in range(5001):
        _,f = cap.read()

    # Skip rows before starting object detection
    
    offset_v = 267
    start_time = time.time()
    obj_mask, obj_x, obj_y = object_mask(f, bg, offset_v)
    print "Object mask completed in", (time.time() - start_time), "s"
    print "Object points:", obj_x.shape, obj_y.shape
    
    purple = (217, 156, 177)
    fg = mask_background(f, obj_mask, purple)
    
    show_images(f, fg)

    start_time = time.time()
    fvs, fvs_raw = feature_vector(f, obj_x, obj_y, x_max=fheight, y_max=fwidth, increment=4)
    print ("Feature vectors %s (%ds)" % (str(fvs.shape), (time.time() - start_time)))
    np.savetxt("foreground.txt", fvs, '%5.8f')  
    np.savetxt("foreground_unnormalized.txt", fvs_raw, '%5.0f')

    cv2.destroyAllWindows()
    cap.release()

main()
print "Done!"
