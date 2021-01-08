# File to test OpenCV functionality, run filters on video and display it

import cv2 as cv
import numpy as np

import utils
import filters
from edge_detectors import *



# START -> Area for precomputed values that have costly recomputation if done per frame

log = filters.lap_of_gauss(2)

# END


cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # START -> Area for running algorithms on top of the video collected from the Video input device

    # Convert to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # gray = cv.resize(gray, (320,240))
    gray = canny.canny_detector(gray, 4, 8)

    # END

    # Display the resulting frame
    cv.imshow('frame',gray)
    if cv.waitKey(1) == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv.destroyAllWindows()