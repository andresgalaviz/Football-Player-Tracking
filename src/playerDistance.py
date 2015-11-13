import cv2
import cv2.cv as cv
import numpy as np
import math
import sideline as sl

videoName = '..//vid//panorama.avi'
recordFile = '..//txt//playerMovingDistance.txt'
playerInfo = '..//img//playersInfo.jpg'

numOfPlayers = 0
fieldLen = 105.0
fieldWid = 68.0


def getCorresponding(point):
    lu, ll, ru, rl = sl.getCorners()

    orinX = point[0]
    orinY = point[1]
    lux = lu[0]
    llx = ll[0]
    rux = ru[0]
    rlx = rl[0]
    far = lu[1]
    near = ll[1]
    nearLen = rlx - llx
    farLen = rux - lux
    width = near - far

    def getXonLeft(y):
        return int(float(y - near) / float(far - near) * float(lux - llx)) + llx

    def getXonRight(y):
        return rlx - int(float(y - near) / float(far - near) * float(rlx - rux))

    xl = getXonLeft(orinY)
    xr = getXonRight(orinY)

    xRatio = float(orinX - xl) / float(xr - xl)

    xf = xRatio * farLen + lux
    xn = xRatio * nearLen + llx
    tempx = float(orinX - xf)
    tempy = float(orinY - far)
    tempnearx = float(xn - xf)

    YRatio = math.sqrt((tempx * tempx + tempy * tempy) / (tempnearx * tempnearx + width * width))

    RealX = xRatio * fieldLen
    RealY = YRatio * fieldWid
    return (RealX, RealY)

def getDistance(point1, point2):
    correspondingPoint1 = getCorresponding(point1)
    correspondingPoint2 = getCorresponding(point2)
    p1x = correspondingPoint1[0]
    p1y = correspondingPoint1[1]
    p2x = correspondingPoint2[0]
    p2y = correspondingPoint2[1]
    return math.sqrt((p1x - p2x) * (p1x - p2x) + (p1y - p2y) * (p1y - p2y))

# get the player list
# to be implemented
# use cv.GoodFeaturesToTrack to get fake players first
def getplayers(frame):
    return cv.GoodFeaturesToTrack(frame, None, None, numOfPlayers, 0.1, 1)


# use fake points for testing
def fakePlayer():
    return [(165,105), (10, 53)]

def compute(playerList, video):
    videoName = video
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

    numOfPlayers = len(playerList)

    # store players moving distance
    players = np.zeros(numOfPlayers)

    # store players position of last frame
    prePlayers = playerList 
    # store players position of current frame
    curPlayers = [] 

    img = cv.CreateImage((width,height), 8, 1)

    #flag of storing player info
    flagInfo = True

    for f in xrange(count):
        frame = cv.QueryFrame(capture)

        if(flagInfo):
            cv.CvtColor(frame, img, cv.CV_BGR2GRAY)
            for i in range(numOfPlayers):
                font=cv.InitFont(cv.CV_FONT_HERSHEY_SCRIPT_SIMPLEX, 0.4, 0.4, 0, 2, 3)
                
                cv.PutText(img, str(i), (int(prePlayers[i][0][0]), int(prePlayers[i][0][1])), font, (255,255,255))
            cv.SaveImage(playerInfo,img)
            flagInfo = False

        
        #Convert to gray
        cv.CvtColor(frame, curFrame, cv.CV_BGR2GRAY) 
        
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
    i = 0
    f = open(recordFile, 'w')
    for player in players:
        i += 1
        print "player", i, "running distance: ", player, "\n"
        f.write("player" + str(i) +" running distance: " + str(player) + "meters\n")

###compute(fakePlayer(), videoName)

