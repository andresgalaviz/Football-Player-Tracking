import cv2
import cv2.cv as cv
import numpy as np

videoName = '..//vid//panorama.mov' # video
writeVedioName = '..//vid//offside.avi'
bgName = '..//img//background.jpg' #background picture
color = (123,123,255) # line color
videoWriter = cv2.VideoWriter(writeVedioName, -1, 0, (0,0))

#get left upper corner, left lower corner, right upper corner and right lower corner position
# to be implemented, use fake field now
def getCorners():
    lu, ll, ru, rl =(1059,52), (132,314),(1675,49),(2563,261)
    return lu, ll, ru, rl

# check whether player outside upper line
def checkUpperArea(point, lu, ll, ru, rl):
    return point[1] < lu[1]

# check whether player outside lower line
def checkLowerArea(point, lu, ll, ru, rl):
    return point[1] > ll[1]

# check whether player outside left line
def checkLeftArea(point, lu, ll, ru, rl):
    if(point[0] > lu[0]):
        return False
    else:
        return (float(lu[0] - point[0]) / float(lu[0] - ll[0])) > (float(lu[1] - point[1]) / float(lu[1] - ll[1]))

# check whether player outside right line
def checkRightArea(point, lu, ll, ru, rl):
    if(point[0] < ru[0]):
        return False
    else:
        return (float(point[0] - ru[0]) / float(rl[0] - ru[0])) > (float(ru[1] - point[1]) / float(ru[1] - rl[1]))

# draw a line on the upper sideline
def drawUpperLine(point, lu, ll, ru, rl):
    y = lu[1]
    x1 = point[0] - 5
    x2 = point[0] + 5
    if(x2 < lu[0]):
    	x2 = lu[0]
    if(x1 > ru[0]):
    	x1 = ru[0]
    return x1, y, x2, y

# draw a line on the lower sideline
def drawLowerLine(point, lu, ll, ru, rl):
    y = ll[1]
    x1 = point[0] - 5
    x2 = point[0] + 5
    if(x2 < ll[0]):
    	x2 = ll[0]
    if(x1 > rl[0]):
    	x1 = rl[0]
    return x1, y, x2, y

# draw a line on the left sideline
def drawLeftLine(point, lu, ll, ru, rl):
    y1 = point[1] - 5
    y2 = point[1] + 5
    def getX(y):
    	return int(float(y - ll[1]) / float(lu[1] - ll[1]) * float(lu[0] - ll[0])) + ll[0]

    x1 = getX(y1)
    x2 = getX(y2)
    return x1, y1, x2, y2

# draw a line on the right sideline
def drawRightLine(point, lu, ll, ru, rl):
    y1 = point[1] - 5
    y2 = point[1] + 5
    def getX(y):
    	return rl[0] - int(float(y - rl[1]) / float(ru[1] - rl[1]) * float(rl[0] - ru[0]))

    x1 = getX(y1)
    x2 = getX(y2)
    return x1, y1, x2, y2

#if player is outside, draw a line at the corresponding sideline. 
#If player is outside a corner, treat it as a point outside the upper/ lower sideline
def checkOutSide(frame, point, lu, ll, ru, rl):
    flag = False
    if(checkUpperArea(point, lu, ll, ru, rl)):
        flag = True
        x1, y1, x2, y2 = drawUpperLine(point, lu, ll, ru, rl)

    elif(checkLowerArea(point, lu, ll, ru, rl)):
        flag = True
        x1, y1, x2, y2 = drawLowerLine(point, lu, ll, ru, rl)

    elif(checkLeftArea(point, lu, ll, ru, rl)):
        flag = True
        x1, y1, x2, y2 = drawLeftLine(point, lu, ll, ru, rl)

    elif(checkRightArea(point, lu, ll, ru, rl)):
        flag = True
        x1, y1, x2, y2 = drawRightLine(point, lu, ll, ru, rl)

    if(flag):
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.line(frame,p1,p2,color, 2)

    return frame

# get the player list
# to be implemented
# use cv.GoodFeaturesToTrack to get fake players first
def extractPlayers(frame):
    #Convert to gray
    cv.CvtColor(frame, frame, cv.CV_BGR2GRAY) 
    return cv.GoodFeaturesToTrack(frame, None, None, 22, 0.1, 1)

#fake positions for testing
def fakePlayers():
    return [(1,1), (95,95), (1,60), (350,60), (10,1), (40,1), (400,95)]

def main():
    cap = cv2.VideoCapture(videoName)
    width = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CV_CAP_PROP_FPS))
    count = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))
    videoWriter = cv2.VideoWriter(writeVedioName, -1, fps, (width, height))

    lu, ll, ru, rl = getCorners(bgName)
    
    for f in xrange(count):
        _, frame = cap.read()
	#playerList = extractPlayers(frame)
        playerList = fakePlayers()
        for player in playerList:
            frame = checkOutSide(frame, player)
        videoWriter.write(frame)
        #####cv2.imshow("test", frame)
        #####cv2.waitKey(100)
        #####print "sth happens"    
    cv2.destroyAllWindows()
    print "end"

def drawLine(frame, players_pos):
    lu, ll, ru, rl = getCorners()
    for player in players_pos:
            newFrame = checkOutSide(frame, player, lu, ll, ru, rl)
    return newFrame