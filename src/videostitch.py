import os
import cv2
import cv2.cv as cv
import numpy as np
from matplotlib import pyplot as plt

def findHomoMatrix(img1, img2):
    sift=cv2.SIFT()
    point1, descriptor1=sift.detectAndCompute(img1,None)
    point2, descriptor2 =sift.detectAndCompute(img2,None)

    flannIndexKDTree = 0
    indexParam = {'algorithm': flannIndexKDTree, 'trees': 5}
    searchParam = {'checks': 50}

    flann = cv2.FlannBasedMatcher(indexParam, searchParam)

    matches = flann.knnMatch(descriptor1,descriptor2,k=2)

    # keep good matches of points
    goodMatches = []
    for x,y in matches:
        if x.distance < 0.7*y.distance:
            goodMatches.append(x)

    srcPoints = np.float32([ point1[x.queryIdx].pt for x in goodMatches ]).reshape(-1,1,2)#cordinates of the points
    dstPoints = np.float32([ point2[x.trainIdx].pt for x in goodMatches ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC,5.0)
    print('homography matrix found:')
    print(M)
    return M

def warpTwoImages(img1, img2, H):
    '''warp img2 to img1 with homograph matrix H'''
    height1,width1 = img1.shape[:2]
    height2,width2 = img2.shape[:2]
    p1 = np.float32([[0,0],[0,height1],[width1,height1],[width1,0]]).reshape(-1,1,2)
    p2 = np.float32([[0,0],[0,height2],[width2,height2],[width2,0]]).reshape(-1,1,2)
    p3 = cv2.perspectiveTransform(p2, H)

    #combint two images and reshape the image size
    points = np.concatenate((p1, p3), axis=0)
    [xmin, ymin] = np.int32(points.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(points.max(axis=0).ravel() + 0.5)
    M = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]])

    outputImg = cv2.warpPerspective(img2, M.dot(H), (xmax-xmin, ymax-ymin))
    outputImg[-ymin:height1-ymin,-xmin:width1-xmin] = img1
    return outputImg


#step1: capture the video and get the frames
cap1=cv2.VideoCapture('..//vid//football_left.mp4')
cap2=cv2.VideoCapture('..//vid//football_mid.mp4')
cap3 = cv2.VideoCapture('..//vid//football_right.mp4')

frameCount=int(cap1.get(cv.CV_CAP_PROP_FRAME_COUNT))


#stitch the first 3 frames, get homography matrices used in the remaining frames
if not cap1.isOpened():
    print('cannot open video 1')

#step 1: capture frame
ret,img1=cap1.read()
ret,img2=cap2.read()
ret,img3=cap3.read()

###################################################
#step2: find SIFT features in left and mid images and apply the ratio test to find the best matches, and find homography matrix
M1=findHomoMatrix(img1,img2)

#step3: stitch left and mid images
leftAndMid=warpTwoImages(img2,img1,M1)

#crop the resulting image because its out of the field, not of our interest
width1=len(leftAndMid)
length1=len(leftAndMid[1,:])
img4=leftAndMid[100:(5*width1/9),(length1/10):]#remove unnecessary edges of the frames.

#step 4: find homography matrix for left image and rightAndMid, and do the stitching.
M2=findHomoMatrix(img3,img4)
panorama=warpTwoImages(img4,img3,M2)

#step5: crop the panorama image discard unintersting pixels
width2 = len(panorama)
length2 = len(panorama[1,:])
panorama=panorama[(width2/8):(4*width2/7),1:19*length2/20]#crop the picture because otherwise it will be too big for later process.

#step6: write video
panorama=cv2.resize(panorama,(0,0),fx=0.4,fy=0.4)

height,width,layer=panorama.shape

fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') # note the lower case
video = cv2.VideoWriter('..//vid//panorama.mov',fourcc,fps=24,frameSize=(width,height),isColor=1)

video.write(panorama)

#do the same to all the remaining frames
for x in range(0, frameCount-1):

    if not cap1.isOpened():
        print('cannot open video 1')

    ret,img1=cap1.read()
    ret,img2=cap2.read()
    ret,img3=cap3.read()

    leftAndMid=warpTwoImages(img2,img1,M1)

    img4=leftAndMid[100:(5*width1/9),(length1/10):]#img4=rightAndMid[(width/8):(7*width/13),1:(17*length/20)]

    panorama=warpTwoImages(img4,img3,M2)

    panorama=panorama[(width2/8):(4*width2/7),1:19*length2/20]
    panorama=cv2.resize(panorama,(0,0),fx=0.4,fy=0.4)
    video.write(panorama)

