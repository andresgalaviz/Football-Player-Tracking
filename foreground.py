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
    hue_threshold = int(0.15 * 180)
    sat_threshold = int(0.15 * 256)
    hue_threshold_sq = hue_threshold**2
    sat_threshold_sq = sat_threshold**2
    n_rows = img.shape[0]
    n_cols = img.shape[1]
    mask = np.ones([n_rows, n_cols], dtype=np.int)
    bg_hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    obj_x = []
    obj_y = []
    
    for i in range(offset_v - 1, n_rows - 1):
        for j in range(offset_h - 1, n_cols - 1):
            hue_diff = int(img_hsv[i, j, 0]) - int(bg_hsv[i, j, 0])
            hue_diff_sq = hue_diff**2
            if (hue_diff_sq > hue_threshold_sq):
                mask[i, j] = 0
                obj_x.append(i)
                obj_y.append(j)
                continue
            sat_diff = int(img_hsv[i, j, 1]) - int(bg_hsv[i, j, 1])
            sat_diff_sq = sat_diff**2
            if (sat_diff_sq > sat_threshold_sq):
                mask[i, j] = 0
                obj_x.append(i)
                obj_y.append(j)
                continue
    return mask, np.array(obj_x, dtype=np.int), np.array(obj_y, dtype=np.int)

# Mask background with given color (default: black)

def mask_background(img, mask, mask_color=(0, 0, 0)):
    result = np.array(img, copy=True)
    result[mask == 1] = mask_color
    return result

# Features comprises:
# x coord (row index)
# y coord (column index)
# hue value
# saturation value

def feature_vector(img, obj_x, obj_y, x_max, y_max):
    n_features = 4
    result = np.zeros([1, n_features])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    n_points = obj_x.size
    for i in range(0, n_points - 1):

        # Normalize the values
        
        x = obj_x[i] / (x_max - 1)
        y = obj_y[i] / (y_max - 1)
        hue = hsv[x, y, 0] / 179.
        sat = hsv[x, y, 1] / 255.
        fv = np.array([x, y, hue, sat])
        result = np.vstack((result, fv))
    result = np.delete(result, 0, 0)
    return result

#

def cluster(data):
    _, classified_points, centroids = cv2.kmeans(data=data, K=18, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10), attempts=1, flags=cv2.KMEANS_RANDOM_CENTERS)
    return centroids


def main():

    bg = cv2.imread("background.jpg", cv2.CV_LOAD_IMAGE_COLOR)
    
    # Read the video

    cap = cv2.VideoCapture("football_right.mp4")
    n_frames = cap.get(cv.CV_CAP_PROP_FRAME_COUNT)
    fwidth = cap.get(cv.CV_CAP_PROP_FRAME_WIDTH)
    fheight = cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv.CV_CAP_PROP_FPS)
    
    #print "Frame width: ", fwidth, "\nFrame height: ", fheight, "\nFrames per second: ", fps, "\nFrame count: ", fcount

    

    _,f = cap.read()

    # Skip rows before starting object detection
    
    offset_v = 211
    start_time = time.time()
    obj_mask, obj_x, obj_y = object_mask(f, bg, offset_v)
    print "Object mask completed in", (time.time() - start_time), "s"
    
    
    
    fg = mask_background(f, obj_mask)
    
    #cv2.imshow("original", f)
    #cv2.imshow("foreground", fg)
    #cv2.waitKey(0)

    #start_time = time.time()
    #fvs = feature_vector(f, obj_x, obj_y, x_max=fheight, y_max=fwidth)
    #print "Feature vectors completed in", (time.time() - start_time), "s"
    #np.savetxt("fvs.txt", fvs, '%5.8f')

    fvs = np.loadtxt("fvs.txt")
    fvs = fvs.astype(float)

    start_time = time.time()
    centroids = cluster(fvs)
    print "Clustering completed in", (time.time() - start_time), "s"
    np.savetxt("centroid.txt", centroids, '%5.8f')
    
    cv2.destroyAllWindows()
    cap.release()

main()
print "Done!"
