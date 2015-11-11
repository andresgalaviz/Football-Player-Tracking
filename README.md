# Football-Player-Tracking
Soccer player tracking system

Please read completely:

To execute system you must run src/main.py. 

Background extraction will not be executed as we have provided a pre computed background however you can test this by deleting the background dile 'img/bg.jpg'. This will extract the background from the first 5000(or length of video) frames, whichever is smaller. The video is HD quality so it will take time.

We have provided a homography matrix that has been calculated and calibrated for our system but you can test the homography creation by renaming the file 'txt/hgmatrix.txt', after this if you run the system it will ask you to calibrate the points by selecting them from the extracted background.

We have also provided the stitched video and player points for our system which have been pre-calculated. Again, if you want to test this you can simply rename the files and the system will re calculate everything. It will take time. 
