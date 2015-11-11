import os
import cv2
import cv2.cv as cv
import numpy as np
from matplotlib import pyplot as plt
os.chdir("D:/rongliWork/")

def findHomoMatrix(img1, img2):
    MIN_MATCH_COUNT=10
    sift=cv2.SIFT()
    kp1, des1=sift.detectAndCompute(img1,None)
    kp2, des2 =sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    #print(len(good))

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    print('homography matrix found:')
    print(M)
    return M

def warpTwoImages(img1, img2, H):
    '''warp img2 to img1 with homograph H'''
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    return result

def imageStitch(img1, img2, img3):
    # cv2.imwrite('left.jpg',img1)
    # cv2.imwrite('mid.jpg',img2)
    # cv2.imwrite('right.jpg',img3)

    #step2: find SIFT features in left and mid images and apply the ratio test to find the best matches, and find homography matrix
    M1=findHomoMatrix(img1,img2)

    #step3: stitch left and mid images
    leftAndMid=warpTwoImages(img2,img1,M1)

    #crop the resulting image because its out of the field, not of our interest
    width=len(leftAndMid)
    length=len(leftAndMid[1,:])
    #img4=rightAndMid[(width/8):(7*width/13),1:(17*length/20)]
    img4=leftAndMid[100:(5*width/9),(length/10):]

    #step 4: find homography matrix for left image and rightAndMid, and do the stitching.
    #cv2.imwrite('leftAndMid.jpg',img4)
    M2=findHomoMatrix(img3,img4)
    panorama=warpTwoImages(img4,img3,M2)

    #step5: crop the panorama image discard unintersting pixels
    #print(panorama.shape)
    width = len(panorama)
    length = len(panorama[1,:])
    print(panorama.shape)
    panorama=panorama[(width/8):(4*width/7),1:19*length/20]
    print(panorama.shape)
    return(panorama)

#step1: capture the video and get the frames
cap1=cv2.VideoCapture('football_left.mp4')
cap2=cv2.VideoCapture('football_mid.mp4')
cap3 = cv2.VideoCapture('football_right.mp4')

frameCount=cap1.get(cv.CV_CAP_PROP_FRAME_COUNT)

#cv.CreateVideoWriter('panorama.mp4', cv.CV_FOURCC('F', 'M', 'P', '4'),20, (1230,9260), is_color=1)
#height=1230
#width=9260

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
img4=leftAndMid[100:(5*width1/9),(length1/10):]#img4=rightAndMid[(width/8):(7*width/13),1:(17*length/20)]

#step 4: find homography matrix for left image and rightAndMid, and do the stitching.
#cv2.imwrite('leftAndMid.jpg',img4)
M2=findHomoMatrix(img3,img4)
panorama=warpTwoImages(img4,img3,M2)

#step5: crop the panorama image discard unintersting pixels
width2 = len(panorama)
length2 = len(panorama[1,:])
panorama=panorama[(width2/8):(4*width2/7),1:19*length2/20]
#print(panorama.shape)

#step6: write video
panorama=cv2.resize(panorama,(0,0),fx=0.4,fy=0.4)
#cv2.imwrite('firstPanorama.jpg',panorama)
#######################################
#panorama=cv2.imread(filename='firstPanorama.jpg')
height,width,layer=panorama.shape
#video = cv2.VideoWriter('panorama.mp4',fourcc=cv2.cv.CV_FOURCC('F','M','P','4'),fps=24,frameSize=(width,height),isColor=1)
video = cv2.VideoWriter('panoramaFull1.avi',fourcc=-1,fps=24,frameSize=(width,height),isColor=1)

video.write(panorama)

#do the same to all the 7199 remaining frames 
for x in range(0, 7199):

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

    print(x)













