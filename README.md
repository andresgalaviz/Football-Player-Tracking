# Football-Player-Tracking
Soccer player tracking system

Please read completely:

To execute system you must run src/main.py. 

Background extraction will not be executed as we have provided a pre computed background however you can test this by deleting the background dile 'img/bg.jpg'. This will extract the background from the first 5000(or length of video) frames, whichever is smaller. The video is HD quality so it will take time.

We have provided a homography matrix that has been calculated and calibrated for our system but you can test the homography creation by renaming the file 'txt/hgmatrix.txt', after this if you run the system it will ask you to calibrate the points by selecting them from the extracted background.

We have also provided the stitched video and player points for our system which have been pre-calculated. Again, if you want to test this you can simply rename the files and the system will re calculate everything. It will take time. 

## Object Detection

### Method 1 - Clustering

Key Ideas
- Define player objects by clustering points by position and colour (hue)
- Get feet of player by taking the lowest point in the cluster (highest x-coordinate)
- Identify player team by comparing average hue of cluster

Scripts should be run in the following order:
1. foreground.py
  1. input: video frame, reference background image.
  1. Output
    1. foreground image
    1. normalized feature vectors (scaled to 0 - 1, "foreground.txt")
    1. unnormalized feature vectors (preserving x,y coordinates for marking feet later, "foreground_unnormalized.txt")
  1. Identifies an object point in HSV colour space - if hue difference between background and foreground point, then foreground point is an object point.
  1. Reduce noise points outside the field - only when background point matches green field, then consider the point for object detection.
1. distance.py
  1. Calculate euclidean distance of feature vectors, generate m x m distance matrix.
  1. Input: normalized feature vectors
  1. Output: "distance.txt.gz"
1. dbscan.py
  1. Perform dbscan clustering, mark foot of clusters on frame. 
  1. Input: "distance.txt.gz"
  1. Output: "dbscan.txt" is the cluster ID assignment for each object point --> array index refers to the index of the object point in "foreground_unnormalized.txt"
  1. Lastly "dbscan-clusters.jpg" showing the result.
  1. Decided to use density-based clustering like DBSCAN because k-medoid is unstable and did not give good results.
