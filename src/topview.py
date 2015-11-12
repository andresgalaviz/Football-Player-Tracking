import numpy as np
import numpy.linalg as la
import cv2
import cv2.cv as cv
import math
import matplotlib.pyplot as plt
import time

# Count runtime

start_time = time.time()
field_corners = np.empty([4,2], dtype = "float32")
field_counter = 0 

player_corners = np.empty([4,2], dtype = "float32")
player_counter = 0 

hgcoord_filepath = '..//txt//hgmatrix.txt'

def field_click(event, x, y, flags, param):
	global field_counter
	if (event == cv.CV_EVENT_LBUTTONUP):
		if (field_counter >=4):
			print "Press any key to continue"
		else:
			field_corners[field_counter, :] = [x,y]
			print x,y
			field_counter +=1

def player_click(event, x, y, flags, param):
	global player_counter
	if (event == cv.CV_EVENT_LBUTTONUP):
		if (player_counter >=4):
			print "Press any key to continue"
		else:
			player_corners[player_counter, :] = [x,y]
			print x,y
			player_counter +=1

def create_homography():
	global field_counter
	filename_topview = '..//img//top-view.jpg'
	filename_sideview = '..//img//side-view.png'
	
	top_image = cv2.imread(filename_topview)
	side_image = cv2.imread(filename_sideview)
	print "Select the transformation points from the image"
	cv2.namedWindow('Player-View')
	cv.SetMouseCallback('Player-View', player_click, None)
	cv2.imshow('Player-View', side_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	pts = np.matrix(np.zeros(shape=(len(player_corners),3)))
	c = 0
	for i in player_corners:
	    x,y = i.ravel()
	    
	    pts[c,:] = np.array([x,y,1], dtype = "float32")
	    cv2.circle(side_image,(x,y),3,255,-1)
	    c+=1
	
	print "Select the four corners from the Background"
	cv2.namedWindow('Side-View')
	cv.SetMouseCallback('Side-View', field_click, None)
	cv2.imshow('Side-View', side_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	cv2.imshow('Points of interest', side_image)
	side_view_corners = np.copy(field_corners)
	
	top_view_corners = np.array([[44, 393], [44, 30], [598,30], [598, 393]], dtype  = "float32")

	H = cv2.findHomography(side_view_corners, top_view_corners)[0]

	cv2.waitKey(0)
	newPoints = np.empty([1,3], dtype = "float32")
	
	for i in pts:
		newPoints = H*(i.T)
		x = newPoints[0]/float(newPoints[2])
		y = newPoints[1]/float(newPoints[2])
		cv2.circle(top_image,(x,y),3,255,-1)
	cv2.imshow('Birds eye', top_image)
	cv2.waitKey(0)
	np.savetxt(hgcoord_filepath, H)