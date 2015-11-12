import cv2
import cv2.cv as cv
import numpy as np

videoName = 'panoramaFull1.avi' # video
bgName = 'background.jpg' #background picture
color = (255,255,255) # line color

#get left upper corner, left lower corner, right upper corner and right lower corner position
# ro be implemented
def getCorners(backGroundPic):
	lu, ll, ru, rl
	return lu, ll, ru, rl

lu, ll, ru, rl = getCorners(bgName)

# check whether player outside upper line
def checkUpperArea(point):
	return point[1] < ll[1]

# check whether player outside lower line
def checkLowerArea(point):
	return point[1] > ll[1]

# check whether player outside left line
def checkLeftArea(point):
	if(point[0] > lu[0]):
		return false

	return (float(lu[0] - point[0]) / float(lu[0] - ll[0])) > (float(lu[1] - point[1]) / float(lu[1] - ll[1]))

# check whether player outside right line
def checkRightArea(point):
	if(point[0] < ru[0]):
		return false

	return (float(point[0] - ru[0]) / float(rl[0] - ru[0])) > (float(ru[1] - point[1]) / float(ru[1] - rl[1]))

# draw a line on the upper sideline
def drawUpperLine(frame, point):
	y = lu[1]
	x1 = point[0] - 5
	x2 = point[0] + 5
	if(x2 < lu[0]):
		x2 = lu[0]
	if(x1 > ru[0]):
		x1 = ru[0]
	cv.Line(frame, [x1,y], [x2,y], color)

# draw a line on the lower sideline
def drawLowerLine(frame, point):
	y = ll[1]
	x1 = point[0] - 5
	x2 = point[0] + 5
	if(x2 < ll[0]):
		x2 = ll[0]
	if(x1 > rl[0]):
		x1 = rl[0]
	cv.Line(frame, [x1,y], [x2,y], color)

# draw a line on the left sideline
def drawLeftLine(frame, point):
	y1 = point[1] - 5
	y2 = point[1] + 5
	def getX(y):
		return int(float(y - ll[1]) / float(lu[1] - ll[1]) * float(lu[0] - ll[0])) + ll[0]

	x1 = getX(y1)
	x2 = getX(y2)
	cv.Line(frame, [x1,y], [x2,y], color)

# draw a line on the right sideline
def drawRightLine(frame, point):
	y1 = point[1] - 5
	y2 = point[1] + 5
	def getX(y):
		return rl[0] - int(float(y - rl[1]) / float(ru[1] - rl[1]) * float(rl[0] - ru[0]))

	x1 = getX(y1)
	x2 = getX(y2)
	cv.Line(frame, [x1,y], [x2,y], color)

#if player is outside, draw a line at the corresponding sideline. 
#If player is outside a corner, treat it as a point outside the upper/ lower sideline
def checkOutSide(frame, point):
	if(checkUpperArea(point)):
		drawUpperLine(frame, point)

	if(checkLowerArea(point)):
		drawLowerLine(frame, point)

	if(checkLeftArea(point)):
		drawLeftLine(frame, point)

	if(checkRightArea(point)):
		drawRightLine(frame, point)

# get the player list
# to be implemented
# use cv.GoodFeaturesToTrack to get fake players first
def extractPlayers(frame):
    #Convert to gray
    cv.CvtColor(frame, curFrame, cv.CV_BGR2GRAY) 
	return prePlayers = cv.GoodFeaturesToTrack(frame, None, None, 22, 0.1, 1)

def main():
	cap = cv2.VideoCapture(videoName)
	width = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
	fps = int(cap.get(cv.CV_CAP_PROP_FPS))
	count = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))

	for f in xrange(nbFrames):
		frame = cv.QueryFrame(capture)
		playerList = extractPlayers
		for player in playerList:
			checkOutSide(frame, player)



