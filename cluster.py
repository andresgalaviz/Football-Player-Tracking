import time
import numpy as np
import random
import cv2
import cv2.cv as cv

def cluster(distances, k=3):

    m = distances.shape[0] # number of points

    # Pick k random medoids.
    curr_medoids = np.array([-1]*k)
    while not len(np.unique(curr_medoids)) == k:
        curr_medoids = np.array([random.randint(m/4, 3*m/4) for _ in range(k)])
        #curr_medoids = np.array([random.randint(0, m - 1) for _ in range(k)])
    old_medoids = np.array([-1]*k) # Doesn't matter what we initialize these to.
    new_medoids = np.array([-1]*k)
   
    # Until the medoids stop updating, do the following:
    while not ((old_medoids == curr_medoids).all()):
        # Assign each point to cluster with closest medoid.
        clusters = assign_points_to_clusters(curr_medoids, distances)

        # Update cluster medoids to be lowest cost point. 
        for curr_medoid in curr_medoids:
            cluster = np.where(clusters == curr_medoid)[0]
            new_medoids[curr_medoids == curr_medoid] = compute_new_medoid(cluster, distances)

        old_medoids[:] = curr_medoids[:]
        curr_medoids[:] = new_medoids[:]

    return clusters, curr_medoids

def assign_points_to_clusters(medoids, distances):
    distances_to_medoids = distances[:,medoids]
    clusters = medoids[np.argmin(distances_to_medoids, axis=1)]
    clusters[medoids] = medoids
    return clusters

def compute_new_medoid(cluster, distances):
    mask = np.ones(distances.shape)
    mask[np.ix_(cluster,cluster)] = 0.
    cluster_distances = np.ma.masked_array(data=distances, mask=mask, fill_value=10e9)
    costs = cluster_distances.sum(axis=1)
    return costs.argmin(axis=0, fill_value=10e9)   

def show_medoids_on_frame(medoids):
    pts = np.loadtxt("foreground_unnormalized.txt", dtype=int)
    cap = cv2.VideoCapture("football_right.mp4")
    _,f = cap.read()
    red = (0, 0, 255)
    blue = (255, 0, 0)
    cv2.imshow("Original", f)
    cv2.imwrite("cluster-input.jpg", f)
    for pt in pts:
        cv2.circle(f, (int(pt[1]),int(pt[0])), 1, blue,thickness=-1)
    for medoid in medoids:
        cv2.circle(f, (int(pts[medoid, 1]),int(pts[medoid, 0])), 10, red,thickness=1)
    cv2.imshow("Medoids", f)
    cv2.imwrite("cluster-medoid.jpg", f)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()
    

def main():    
    n_clusters = int(18*2)
    distances = np.loadtxt("distance.txt.gz")
    print "distances:", distances.shape
    
    start_time = time.time()
    clusters, medoids = cluster(distances, k=n_clusters)
    print ("Clustering k=%d (%ds)" % (n_clusters, time.time() - start_time))

    np.savetxt("cluster.txt", clusters, '%5.8f')
    np.savetxt("cluster-medoid.txt", medoids, '%5.8f')

    show_medoids_on_frame(medoids)

main()
print "Done!"
