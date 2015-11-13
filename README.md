# Football-Player-Tracking
Soccer player tracking system

Please read completely:
## Products

### Stitched video - pre detection.
Video has been pre-calculated. It is located as "vid/panorama.mov". Again, if you want to test this you can simply rename the files and the system will re calculate everything. It will take time. 

### Real Time Football Player Tracking
To execute system you must run "python main.py" from inside the src folder. In order to work you must make sure to have the required video codecs(Very important in Mac). If you are taking the code from this repository you must also follow the README.md file inside the "vid/" folder.

The system will do the following:
#### Background extraction
*Background extraction will not be executed as we have provided a pre computed background however you can test this by deleting the background file 'img/side-view.jpg'.* 

Source code: "src/bgextraction.py"
This will extract the background from the video frames by using the averaging technique. The video is HD quality so it will take time. 

### Homography Matrix extraction for topview
*Homography Matrix extraction will not be executed as we have provided a pre computed homography however you can test this by deleting the file 'txt/hgmatrix.txt'.* 

Source code: "src/topview.py" Method create_homography()
This method will use the extracted background and prompt the user to manually select with the pointer the four corners of this image. The image is quite large so it might have to be moved, the quality of the top projection is dependent on how well the corners are selected. The corners should be selected: Left-Down, Left-Top, Right-Top, Right-Down


We have provided a homography matrix that has been calculated and calibrated for our system but you can test the homography creation by renaming the file 'txt/hgmatrix.txt', after this if you run the system it will ask you to calibrate the points by selecting the four corners from the extracted background.



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
