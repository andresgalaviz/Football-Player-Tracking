# Football-Player-Tracking

## Products

### Stitched video
Video has been pre-calculated. It is located under "vid/panorama.mov". Iff you want to test the video stitching you can simply rename the files and the system will re calculate everything. It will take time. 
**The system assumes the video is already created and is located in that location, to test this you mus download the video files Download football_left.mp4, football_mid.mp4, football_right.mp4 from:
https://drive.google.com/folderview?id=0B7gBv2Jut0VxcDNENmxvS2N4Qk0&usp=sharing and place them under the vid folder. Then run "python videostitch.py" from within the "src/" folder.

Source code: "src/videostitch.py"

### Real Time Football Player Tracking
To execute system you must run "python main.py" from inside the src folder. In order to work you must make sure to have the required video codecs(Very important in Mac). If you are taking the code from this repository you must also follow the README.md file inside the "vid/" folder.

The system will do the following:
#### Background extraction
*Background extraction will not be executed as we have provided a pre computed background however you can test this by deleting the background file 'img/side-view.jpg'.* 

Source code: "src/bgextraction.py"

This will extract the background from the video frames by using the averaging technique. The video is HD quality so it will take time. 

### Homography Matrix extraction for Topview
*Homography Matrix extraction will not be executed as we have provided a pre computed homography however you can test this by deleting the file 'txt/hgmatrix.txt'.* 

Source code: "src/topview.py" Method: create_homography()

This method will use the extracted background and prompt the user to manually select with the pointer the four corners of this image. The image is quite large so it might have to be moved, the quality of the top projection is dependent on how well the corners are selected. The corners should be selected: Left-Down, Left-Top, Right-Top, Right-Down. After this the method will calculate a homography based on our provided coordinates for our top image.
**OpenCV Method: cv2.findHomography(side_view_corners, top_view_corners)**

### Real Time Player Tracking
This uses the fairly simple concept of differentiating between background pixel values and frame pixel values. The way this method is implemented is based on the pre asumption that we have a background and that the cameras will not be moving during the duration of the video. 

This function uses several OpenCV methods which were used merely for the means of speed and convenience, all of the methods were covered during the lectures. However while the concepts and techniques were covered in the lectures the specific implementations of the OpenCV methods may be different.

The first step for tracking the players is to get the get the grayscale frame and compare the pixel values to the grayscale background and get the absolute difference between them. This is done with the **OpenCV Method: cv2.absdiff(gray_bg_img, gray_img)**. The next step is to put to 0 values below a certain treshold and to 255 values above a certain treshold. **OpenCV Method: cv2.threshold(bg_delta, 30, 255, cv2.THRESH_BINARY)** Again this is done just for speed and convenience. After this a technique to connect isolated points into one big one is used, it's similar to the ones seen in class where points get dilated by the use of a kernel(In this case the default one) and the result is a bigger united point mostly covering the player or other foreground objects. **OpenCV Method: cv2.dilate(thresholdimage, None, iterations=3)**. Finally a contour of this bigger point is obtained by using the **OpenCV Method: cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)** that uses the technique not covered in class but described in:
*Satoshi Suzuki and others. Topological structural analysis of digitized binary images by border following. Computer Vision, Graphics, and Image Processing, 30(1):32–46, 1985.* This technique basically follows the borders of a binary provided image and alters it to produce contours around the points. 

After this the system loops through each of the detected countours, obtains a bounding rectangle: **OpenCV Method: cv2.boundingRect()** and classifies it as a player with the following criteria:
- Feet point is inside the field polygon
- Contour area is within the limits of players area on x,y location: **OpenCV Method: cv2.contourArea()**
- It has a "human" rectangle-like form
Noise is discarded.

The tracking system then obtains the color of the player by averaging the hue component inside the player rectangle and classifies it as "Red", "Blue", "Unknown". Although the system can classify individual goalkeepers based on color, for the sake of offside measuring they are classified as either blue or red team. Color is drawn to represent the classification outcome.

Source code: "src/playertrack.py" Method: track_player(hg_matrix)

Source code: "src/huematcher.py" Method: average_hue(), is_red_player(), is_blue_player(), is_green_keeper(), is_white_keeper()

Source code: "src/topview.py" 

drawoffside.draw(img, player_top_points) Method: create_topview(hg_matrix, players_pos)



# Proof of concept/Previous approaches: 

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
