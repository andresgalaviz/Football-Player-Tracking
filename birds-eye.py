import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import numpy.linalg as la	
pi = math.pi

#A0145029J_AndresGalavizGomez_Lab7n8

def quatmult(q, r):
	# quaternion multiplication
	out = [0, 0, 0, 0] # output array to hold the result
	out[0] = r[0]*q[0] - r[1]*q[1] - r[2]*q[2] - r[3]*q[3]
	out[1] = r[0]*q[1] + r[1]*q[0] + r[2]*q[3] - r[3]*q[2]
	out[2] = r[0]*q[2] - r[1]*q[3] + r[2]*q[0] + r[3]*q[1]
	out[3] = r[0]*q[3] + r[1]*q[2] - r[2]*q[1] + r[3]*q[0]
	return out

def quat2rot(q):
	rotMat = np.zeros([3, 3])
	rotMat[0,0] = q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3]
	rotMat[0,1] = 2*(q[1]*q[2] - q[0]*q[3])
	rotMat[0,2] = 2*(q[1]*q[3] + q[0]*q[2])
	rotMat[1,0] = 2*(q[1]*q[2] + q[0]*q[3])
	rotMat[1,1] = q[0]*q[0] + q[2]*q[2] - q[1]*q[1] - q[3]*q[3]
	rotMat[1,2] = 2*(q[2]*q[3] - q[0]*q[1])
	rotMat[2,0] = 2*(q[1]*q[3] - q[0]*q[2])
	rotMat[2,1] = 2*(q[2]*q[3] + q[0]*q[1])
	rotMat[2,2] = q[0]*q[0] + q[3]*q[3] - q[1]*q[1] - q[2]*q[2]
	return rotMat

def persProj(pts, camQuat, quatmat):
	u0 =0 
	v0 =0 
	Bu =1 
	Bv =1 
	ku =1 
	kv =1
	f_length = 1
	pts_cam = np.zeros([len(pts), 3])
	for i in range(0,len(pts)):
		u_fp = (f_length * np.dot((pts[i] - camQuat).T, quatmat[0]) * Bu)/(np.dot((pts[i] - camQuat).T, quatmat[2])) + u0
		v_fp = (f_length * np.dot((pts[i] - camQuat).T, quatmat[1]) * Bu)/(np.dot((pts[i] - camQuat).T, quatmat[2])) + v0
		pts_cam[i,:] = [u_fp, v_fp, 1]
	return pts_cam
def ortProj(pts, camQuat, quatmat):
	u0 =0 
	v0 =0 
	Bu =1 
	Bv =1 
	pts_cam = np.zeros([len(pts), 3])
	for i in range(0,len(pts)):
		u_fp = (np.dot((pts[i] - camQuat).T, quatmat[0]) * Bu)
		v_fp = (np.dot((pts[i] - camQuat).T, quatmat[1]) * Bu)
		pts_cam[i,:] = [u_fp, v_fp, 1]
	return pts_cam


filename = 'checkertest.png'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
corners = np.int0(corners)
pts = np.zeros([len(corners), 3])
c = 0
for i in corners:
    x,y = i.ravel()
    pts[c,:] = [x,-1*y + 200,0]
    cv2.circle(img,(x,y),3,255,-1)
    c +=1
plt.imshow(img),plt.show()
for i in range (0,len(pts)):
	plt.scatter(pts[i][0],pts[i][1])

plt.show()

# pts = np.zeros([11, 3])
# pts[0, :] = [0, 0, 0]
# pts[1, :] = [0, 2, 0]
# pts[2, :] = [2, 2, 0]
# pts[3, :] = [4, 2, 0]
# pts[4, :] = [4, 0, 0]
# pts[5, :] = [2, 0, 0]
# pts[6, :] = [1, 1, 0]
# pts[7, :] = [3.5, 1.2, 0]
# pts[8, :] = [2.8, 1.5, 0]
# pts[9, :] = [0.5, 0.5, 0]
# pts[10,:] = [0, 0.5, 0]
for i in range (0,len(pts)):
	plt.scatter(pts[i][0],pts[i][1])
plt.show()
print pts
rotQuat = [math.cos(pi/12), math.sin(pi/12), 0, 0]
camQuat = np.zeros([4, 3])
camQuat[0] = [220, 0, -5]
camQuat[1] = np.dot(quat2rot(rotQuat), camQuat[0])
camQuat[2] = np.dot(quat2rot(rotQuat), camQuat[1])
camQuat[3] = np.dot(quat2rot(rotQuat), camQuat[2])


# Using the 3D shape points defined in 1.1, and the camera translation 
# and orientation for the four frames defined in 1.2 and 1.3, project 
# the 3D shape points onto the image planes for all the four frames. 
# You need to do both the projection models that we have studied in class: orthographic and perspective.
rotQuat = [math.cos(pi/12), math.sin(pi/12), 0, 0]
quatmat = np.zeros([4, 3, 3])
quatmat[0] = [[1,0,0],[0,1,0],[0,0,1]]
quatmat[1] = np.dot(quat2rot(rotQuat), quatmat[0])
quatmat[2] = np.dot(quat2rot(rotQuat), quatmat[1])
quatmat[3] = np.dot(quat2rot(rotQuat), quatmat[2])


fig = plt.figure()
fig.suptitle('Perspective', fontsize=20)
pts_per_third = np.zeros([len(pts), 3])
for i in range (0, 4):
	pts_per = persProj(pts, camQuat[i][:3], quatmat[i])
	if(i == 2):
		pts_per_third = pts_per
	fig.add_subplot(2,2,i+1)   #top left
	for i in range (0,len(pts)):
		
		plt.scatter(pts_per[i][0],pts_per[i][1])


plt.show()

fig = plt.figure()
fig.suptitle('Orthographic', fontsize=20)
for i in range (0, 4):
	pts_ort = ortProj(pts, camQuat[i][:3], quatmat[i])
	fig.add_subplot(2,2,i+1)   #top left
	for i in range (0,len(pts)):
		
		plt.scatter(pts_ort[i][0],pts_ort[i][1])
plt.show()


matrixP = np.zeros([10, 9])

matrixP[0, :] = [pts[0, 0], pts[0, 1], 1, 0, 0, 0, -1*pts_per_third[0,0]*pts[0, 0], -1*pts_per_third[0,0]*pts[0, 1], -1*pts_per_third[0,0]]
matrixP[1, :] = [0, 0, 0, pts[0, 0], pts[0, 1], 1, -1*pts_per_third[0,1]*pts[0, 0], -1*pts_per_third[0,1]*pts[0, 1], -1*pts_per_third[0,1]]


matrixP[2, :] = [pts[1, 0], pts[1, 1], 1, 0, 0, 0, -1*pts_per_third[1,0]*pts[1, 0], -1*pts_per_third[1,0]*pts[1, 1], -1*pts_per_third[1,0]]
matrixP[3, :] = [0, 0, 0, pts[1, 0], pts[1, 1], 1, -1*pts_per_third[1,1]*pts[1, 0], -1*pts_per_third[1,1]*pts[1, 1], -1*pts_per_third[1,1]]


matrixP[4, :] = [pts[2, 0], pts[2, 1], 1, 0, 0, 0, -1*pts_per_third[2,0]*pts[2, 0], -1*pts_per_third[2,0]*pts[2, 1], -1*pts_per_third[2,0]]
matrixP[5, :] = [0, 0, 0, pts[2, 0], pts[2, 1], 1, -1*pts_per_third[2,1]*pts[2, 0], -1*pts_per_third[2,1]*pts[2, 1], -1*pts_per_third[2,1]]


matrixP[6, :] = [pts[3, 0], pts[3, 1], 1, 0, 0, 0, -1*pts_per_third[3,0]*pts[3, 0], -1*pts_per_third[3,0]*pts[3, 1], -1*pts_per_third[3,0]]
matrixP[7, :] = [0, 0, 0, pts[3, 0], pts[3, 1], 1, -1*pts_per_third[3,1]*pts[3, 0], -1*pts_per_third[3,1]*pts[3, 1], -1*pts_per_third[3,1]]


matrixP[8, :] = [pts[8, 0], pts[8, 1], 1, 0, 0, 0, -1*pts_per_third[8,0]*pts[8, 0], -1*pts_per_third[8,0]*pts[8, 1], -1*pts_per_third[8,0]]
matrixP[9, :] = [0, 0, 0, pts[8, 0], pts[8, 1], 1, -1*pts_per_third[8,1]*pts[8, 0], -1*pts_per_third[8,1]*pts[8, 1], -1*pts_per_third[8,1]]

U,S,VT=la.svd(matrixP)

hV = VT[8,:]
homMat = [[hV[0], hV[1], hV[2]], [hV[3], hV[4], hV[5]], [hV[6], hV[7], hV[8]]]
homMat = homMat / hV[8]
print homMat


