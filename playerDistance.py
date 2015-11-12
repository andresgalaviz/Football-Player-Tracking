import cv2
import cv2.cv as cv
import numpy as np
import math

videoName = 'panoramaFull1.avi'
numOfPlayers = 2
def getDistance(point1, point2):
    p1x = point1[0]
    p1y = point1[1]
    p2x = point2[0]
    p2y = point2[1]
    return math.sqrt((p1x - p2x) * (p1x - p2x) + (p1y - p2y) * (p1y - p2y))

# get the player list
# to be implemented
# use cv.GoodFeaturesToTrack to get fake players first
def getplayers(frame):
    return cv.GoodFeaturesToTrack(frame, None, None, numOfPlayers, 0.1, 1)


# use fake points for testing
def fakePlayer():
    return [(165,105), (10, 53)]


def main():
    capture = cv.CaptureFromFile(videoName)

    count = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_COUNT))
    fps = cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FPS)
    width = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT))

    # store the last frame 
    preFrame = cv.CreateImage((width,height), 8, 1) 
    # store the current frame
    curFrame = cv.CreateImage((width,height), 8, 1) 

    prePyr = cv.CreateImage((height / 3, width + 8), 8, cv.CV_8UC1) 
    curPyr = cv.CreateImage((height / 3, width + 8), 8, cv.CV_8UC1) 

    # store players moving distance
    players = np.zeros(numOfPlayers)

    # store players position of last frame
    prePlayers = [] 
    # store players position of current frame
    curPlayers = [] 

    for f in xrange(count):
        frame = cv.QueryFrame(capture)

        # find players
        #Convert to gray
        cv.CvtColor(frame, curFrame, cv.CV_BGR2GRAY) 
        prePlayers = fakePlayer()
        
        #Calculate the movement using the previous and the current frame using the previous points
        curPlayers, status, err = cv.CalcOpticalFlowPyrLK(preFrame, curFrame, prePyr, curPyr, prePlayers, (10, 10), 3, (cv.CV_TERMCRIT_ITER|cv.CV_TERMCRIT_EPS,20, 0.03), 0)

        ###temp = frame
        # add new distance to list
        for i in range(numOfPlayers):
            players[i] += getDistance(prePlayers[i], curPlayers[i])
            ###cv.Line(temp, (int(prePlayers[i][0]), int(prePlayers[i][1])), (int(curPlayers[i][0]), int(curPlayers[i][1])), (255,122,122),3)

        ###cv.ShowImage("test", temp)
        ###cv2.waitKey(20)
        
        #Put the current frame preFrame 
        cv.Copy(curFrame, preFrame) 
        prePlayers = curPlayers
    ###cv2.destroyAllWindows()
    # print distance
    for player in players:
        print "player running distance: ", player, "\n"
main()
