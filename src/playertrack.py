import numpy as np
import cv2
import cv
import time
import datetime
import topview


# def insideField(x,y):
# 	# col: 130, row: 314
# 	# col: 1060, row: 53
# 	# col: 1676, row: 49
# 	# col: 2562, row: 260
# 	# col: 1093, row: 263
bg_filpath = '..//img//side-view.jpg'
vid_filepath = '..//vid//panorama.mov'

def track_player(hg_matrix):
	bg_img = cv2.imread(bg_filpath)
	gray_bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
	vid_cap = cv2.VideoCapture(vid_filepath)
	frame_height = vid_cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT)
	frame_width = vid_cap.get(cv.CV_CAP_PROP_FRAME_WIDTH)
	frame_count = vid_cap.get(cv.CV_CAP_PROP_FRAME_COUNT)
	while True:
		aval, img = vid_cap.read()
		if not aval:
			break
		gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		bg_delta = cv2.absdiff(gray_bg_img, gray_img)

		

		threshold = cv2.threshold(bg_delta, 30, 255, cv2.THRESH_BINARY)[1]
		

		kernel = np.matrix([[0,0,0],[1,1,1],[0,0,0]],np.uint8)
		threshold = cv2.dilate(threshold, None, iterations=3)
		
		contours, _ = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		players_pos = list()
		
		# loop over the contours
		c = 0 
		for cn in contours:
			# if the contour is too small, ignore it
	 		
			# compute the bounding box for the contour, draw it on the frame,
			# and update the text
			(x, y, w, h) = cv2.boundingRect(cn)
			feet_coord = [float(x + int(w/2.0)), float(y + h)]
			
			rect_area = cv2.contourArea(cn)
			if(y > frame_height/4):
				if(rect_area < 40):
					continue
			
			if(w > h*1.4 or rect_area < 10):
				continue

			cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

			players_pos.append(feet_coord)
			c +=1
		top_img = topview.create_topview(hg_matrix, players_pos)
		img=cv2.resize(img,(0,0),fx=0.6,fy=0.6)
		cv2.imshow("Player detection", img)
		cv2.imshow("Top image", top_img)
		key = cv2.waitKey(41) & 0xFF
	vid_cap.release()
	cv2.destroyAllWindows()