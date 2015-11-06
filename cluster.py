import numpy as np
np.set_printoptions(threshold=np.nan)
import numpy.ma as ma
import numpy.linalg as la
import cv2
import cv2.cv as cv
import math
import matplotlib.pyplot as plt
import time
import random

def euclidean_distance(v1, v2):
    sum_sq = 0
    for i in range(v1.shape[0]):
        sum_sq += (v1[i] - v2[i])**2
        #print sum_sq
    return math.sqrt(sum_sq)

# Assign each point to cluster with closest medoid.

def assign_points_to_clusters(medoids, distances):
    #distances_to_medoids = distances[:,medoids]
    #clusters = medoids[np.argmin(distances_to_medoids, axis=1)]
    clusters = medoids[np.argmin(distances, axis=1)]
    #np.savetxt("clusters.txt", clusters, '%5.8f')
    #clusters[medoids] = medoids
    #np.savetxt("clusters2.txt", clusters, '%5.8f')
    return clusters

def distance_to_medoids(points, medoids):
    n_points = points.shape[0]
    n_medoids = medoids.shape[0]
    distances = np.empty([n_points, n_medoids])
    for i in range(n_points):
        pt = points[i,:]
        for j in range(n_medoids):
            medoid = points[medoids[j],:]
            distances[i, j] = euclidean_distance(pt, medoid)
    return distances

def cluster(points, k=2):
    n_points = points.shape[0]

    # Pick k random medoids.
    curr_medoids = np.array([-1]*k)
    while not len(np.unique(curr_medoids)) == k:
        curr_medoids = np.array([random.randint(0, n_points - 1) for _ in range(k)])
    print "Initialize curr_medoids:", curr_medoids

    # Doesn't matter what we initialize these to.

    old_medoids = np.array([-1]*k) 
    new_medoids = np.array([-1]*k)

    start_time = time.time()
    distances = distance_to_medoids(points, curr_medoids)
    print "Computing distance to medoids:", (time.time() - start_time), "s"
    np.savetxt("distances.txt", distances, '%5.8f')

    
    start_time = time.time()
    clusters = assign_points_to_clusters(curr_medoids, distances)
    print "Assigning points to clusters:", (time.time() - start_time), "s"
    np.savetxt("clusters.txt", clusters, '%5.0f')
    

def main():    
    n_clusters = 3
    pts = np.loadtxt("fvs.txt")
    print "points:", pts.shape
    cluster(pts, k=n_clusters)

main()
print "Done!"
