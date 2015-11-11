# Football-Player-Tracking
Soccer player tracking system

Please read completely:

To execute system you must run src/main.py, if no background has been extracted it will extract it from the first 200 frames or length of video, whichever happens first. 

We have provided a homography matrix that has been calculated and calibrated for our system but you can test the homography creation by renaming the file txt/hgmatrix.txt, after this if you run the system it will ask you to calibrate the points by selecting them from the extracted background.

We have also provided the stitched video and player points for our system which have been pre-calculated. Again, if you want to test this you can simply rename the files and the system will re calculate everything. It will take time. 
