import os
import sys
import cv2
import cv2.cv as cv
import numpy as np
# Tested on this video, available at http://www.comp.nus.edu.sg/~ngtk/CS4243Video/football_videos_Android/
# 14k HD frames so be patient...
center_vid_file = 'centre_camera.mp4' 

def backgroundExtraction(videoFile):
	"""Receives a video filename(with extension) and returns the extracted background"""
	vid_cap = cv2.VideoCapture(videoFile)
	if vid_cap.isOpened():
		fps = vid_cap.get(cv.CV_CAP_PROP_FPS)
		frame_height = vid_cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT)
		frame_width = vid_cap.get(cv.CV_CAP_PROP_FRAME_WIDTH)
		frame_count = vid_cap.get(cv.CV_CAP_PROP_FRAME_COUNT)
		print 'FPS', fps
		print 'Frame Height', frame_height
		print 'Frame Width', frame_width
		print 'Frame Count', frame_count

		frame_count = int(frame_count)

		_,img = vid_cap.read()
		avg_img = img

		
		for fr in range(1, frame_count+1):
			_,img = vid_cap.read()
			fr_fl = float(fr)
			avg_img = (fr_fl*avg_img + img)/(fr_fl+1)
			print "Frame = ", fr
			
		avg_img = avg_img
		vid_cap.release()
		return avg_img
	else:
		raise IOError("Could not open video")
	  
def main():
	try:
		
		background = backgroundExtraction(center_vid_file)
		cv2.imshow('Background', background/255)
		cv2.imwrite("Background.jpg", background)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	except IOError as e:
		print "I/O error({0}): {1}".format(e.errno, e.strerror)


main()
