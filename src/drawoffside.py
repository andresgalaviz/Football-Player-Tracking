import cv2.cv as cv
import cv2
import numpy as np

color = (123,123,255)

# get corner position in side view
def getCorners():
	wn = (1059, 52) # west north
	ws = (290, 270) # west south
	en = (1675, 49) # east north
	es = (2563, 261)# east south
	return wn, ws, en, es

# get west and east side position of top view
def getTopViewSide():
	westLine = 42
	eastLine = 597
	return westLine, eastLine

# decide two points to draw a line between them
# input is the offside player position
def decidePoint(point):
	# in top view
	orinX = point[0]
	orinY = point[1]

	# in top view
	westLine, eastLine = getTopViewSide()

	# in side view
	wn, ws, en, es = getCorners()
	wnX = wn[0]
	wnY = wn[1]
	wsX = ws[0]
	wsY = ws[1]
	enX = en[0]
	enY = en[1]
	esX = es[0]
	esY = es[1]
	northLine = (wnY + enY) / 2
	southLine = (wsY + esY) / 2

	# in side view
	northLenth = enX - wnX
	southLenth = esX - wsX

	###def getXonWest(y):
	###	return int(float(y - southLine) / float(northLine - southLine) * float(wnX - wsX)) + wsX

	###def getXonEast(y):
	###	return esX - int(float(y - southLine) / float(northLine - southLine) * float(esX - enX))

	###wX = getXonWest(orinY)
	###eX = getXonEast(orinY)

	###xRatio = float(orinX - wX) / float(eX - wX)

	xRatio = float(orinX - westLine) / float(eastLine - westLine) 
	print xRatio, northLenth, southLenth

	nX = xRatio * northLenth + wnX
	sX = xRatio * southLenth + wsX

	nPoint = (int(nX), int(northLine))
	sPoint = (int(sX), int(southLine))


	return nPoint, sPoint

# add a new player into a list only if neccessary
def addIn(list, newPlayer):
	if(len(list) == 0):
		list.append(newPlayer)
		list.append(newPlayer)
		list.append(newPlayer)
		list.append(newPlayer)
	elif(newPlayer[0] <= list[0][0]):	# < left side first one
		list[1] = list[0]				# first one become second to left
		list[0] = newPlayer				# new one become first to left
	elif(newPlayer[0] <= list[1][0]):	# < left side second one
		list[1] = newPlayer				# new one become second to left
	elif(newPlayer[0] >= list[3][0]):	# > right side first one
		list[2] = list[3]				# first one become second to right
		list[3] = newPlayer				# new one become first to right
	elif(newPlayer[0] >= list[2][0]):	# > right side second one
		list[2] = newPlayer				# new one become second to right

	if(list[0] == list[1]):
		list[1] = newPlayer
	if(list[2] == list[3]):
		list[2] = newPlayer

	return list

# player in structure: [(x, y), 'u']
def draw(img, players):
	bluePlayers = []
	redPlayers = []
	for player in players:
		x = player[0][0]
		y = player[0][1]
		team = player[1]

		if(team == 'b'):
			bluePlayers = addIn(bluePlayers, (x,y))
		elif(team == 'r'):
			redPlayers = addIn(redPlayers, (x,y))

	flag = True
	startPoint = 30
	if(bluePlayers == []):
		font=cv.InitFont(cv.CV_FONT_HERSHEY_COMPLEX, 0.5, 0.8, 0, 1, 5)
		img = cv.fromarray(img)
		cv.PutText(img, 'blue players not enough, need at least 1 blue player', (0,startPoint), font, (0,0,255))
		img = np.asarray(img)
		startPoint += 25
		flag = False
	if(redPlayers == []):
		img = cv.fromarray(img)
		font=cv.InitFont(cv.CV_FONT_HERSHEY_COMPLEX, 0.5, 0.8, 0, 1, 5)
		cv.PutText(img, 'red players not enough, need at least 1 red player', (0,startPoint), font, (0,0,255))
		img = np.asarray(img)
		flag = False

	if(flag):
		if(bluePlayers[0][0] < redPlayers[1][0]):
			p1, p2 = decidePoint(bluePlayers[0])
			cv2.line(img, p1, p2, color, 2)
			###print p1[0], ',,, ', p1[1], '///', p2[0], ',,, ', p2[1]
		if(bluePlayers[2][0] < redPlayers[3][0]):
			p1, p2 = decidePoint(redPlayers[3])
			cv2.line(img, p1, p2, color, 2)
			###print p1[0], ',,, ', p1[1], '///', p2[0], ',,, ', p2[1]

	###for p in bluePlayers:
		###print p[0], ', ', p[1], '-----'
	###print '\n'
	###for p in redPlayers:
		###print p[0], ', ', p[1], '-----'
	###print '\n=======================\n'
	return img

# for testing
def test(index):
	img = cv2.imread('side-view.jpg')
	players1 = [[[94, 50], 'r'], [[144, 50], 'b'], [[194, 50], 'r'], [[244, 50], 'r'], [[398, 50], 'b'], [[448, 50], 'b'], [[498, 50], 'r'], [[548, 50], 'b']] # should have line
	players2 = [[[94, 50], 'b'], [[144, 50], 'r'], [[194, 50], 'r'], [[244, 50], 'r'], [[398, 50], 'b'], [[448, 50], 'b'], [[498, 50], 'b'], [[548, 50], 'r']] # should have line
	players3 = [[[94, 50], 'r'], [[144, 50], 'r'], [[194, 50], 'b'], [[244, 50], 'r'], [[398, 50], 'b'], [[448, 50], 'r'], [[498, 50], 'b'], [[548, 50], 'b']] # should have no line
	players4 = [[[1,1], 'u']]
	if(index == 1):
		img = draw(img, players1)
	elif(index == 2):
		img = draw(img, players2)
	elif(index == 3):
		img = draw(img, players3)
	elif(index == 4):
		img = draw(img, players4)
	img = cv2.resize(img,(0,0),fx=0.6,fy=0.6)
	cv2.imshow('img', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

#test(1)
#test(2)
#test(3) 
#test(4)