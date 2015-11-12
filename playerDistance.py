import cv2
import cv2.cv as cv
import numpy as np

videoName = 'panoramaFull1.avi'

def getDistance(point1, point2):
    return math.sqrt(np.power(point1 - point2, 2).sum())

# get the player list
# to be implemented
# use cv.GoodFeaturesToTrack to get fake players first
def getplayers(frame):
    #Convert to gray
    cv.CvtColor(frame, curFrame, cv.CV_BGR2GRAY) 
    return prePlayers = cv.GoodFeaturesToTrack(frame, None, None, 22, 0.1, 1)
    
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
    players = np.zeros(playerNum)

    # store players position of last frame
    prePlayers = [] 
    # store players position of current frame
    curPlayers = [] 

    for f in xrange(count):
        frame = cv.QueryFrame(capture)
        
        # find players
        prePlayers = getplayers(curFrame) 

        #Calculate the movement using the previous and the current frame using the previous points
        curPlayers, status, err = cv.CalcOpticalFlowPyrLK(preFrame, curFrame, prePyr, curPyr, prePlayers, (10, 10), 3, (cv.CV_TERMCRIT_ITER|cv.CV_TERMCRIT_EPS,20, 0.03), 0)

        # add new distance to list
        for i in range(playerNum):
            players[i] += getDistance(prePlayers[i], curPlayers[i])
           
        #Put the current frame preFrame 
        cv.Copy(curFrame, preFrame) 
        prePlayers = curPlayers

    # print distance
    for player in players:
        print "player running distance: ", player, "\n"
main()