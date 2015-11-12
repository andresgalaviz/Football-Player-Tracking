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
import playertrack

bg_filpath = '..//img//side-view.jpg'
vid_filepath = '..//vid//panorama.mov'
hgmatrix_filepath = '..//txt//hgmatrix.txt'

def main():
	if(not os.path.isfile(bg_filpath)):
		if(not os.path.exists('..//img')):
			os.mkdir('..//img')
		print "Background has not been extracted, will extract."
		bgextraction.extract_background(vid_filepath)
	bg_img = cv2.imread(bg_filpath)
	print "Background image found"
	cv2.imshow("Background Image", bg_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	if(not os.path.isfile(hgmatrix_filepath)):
		if(not os.path.exists('..//txt')):
			os.mkdir('..//txt')
		print "Homography matrix has not been created."
		topview.create_homography()
	hg_matrix = np.loadtxt(hgmatrix_filepath)
	print "Homography matrix found"
	print hg_matrix
	
	playertrack.track_player(hg_matrix)
	
main()