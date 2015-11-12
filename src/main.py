import numpy as np
np.set_printoptions(threshold=np.nan)
import numpy.linalg as la
import os
import os.path
import cv2
import cv2.cv as cv
import math
import matplotlib.pyplot as plt
import time
import bgextraction
import topview

bg_filpath = '..//img//bg.jpg'
vid_filepath = '..//vid//full_field.mp4'
hgmatrix_filepath = '..//txt//hgmatrix.txt'

def main():
	if(not os.path.isfile(bg_filpath)):
		if(not os.path.exists('..//img')):
			os.mkdir('..//img')
		print "Background has not been extracted, will extract."
		bgextraction.extract_background(vid_filepath)
	bg_img = cv2.imread(bg_filpath)

	if(not os.path.isfile(hgmatrix_filepath)):
		if(not os.path.exists('..//txt')):
			os.mkdir('..//txt')
		print "Homography matrix has not been created."
		topview.create_homography()
	hg_matrix = np.loadtxt(hgmatrix_filepath)
	
main()