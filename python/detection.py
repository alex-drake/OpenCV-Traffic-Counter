#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 10:28:47 2017

@author: datascience9
"""

import cv2
import numpy as np
import time
import uuid

# The cutoff for threshold. A lower number means smaller changes between
# the average and current scene are more readily detected.
THRESHOLD_SENSITIVITY = 40
# Number of pixels in each direction to blur the difference between
# average and current scene. This helps make small differences larger
# and more detectable.
BLUR_SIZE = 40
# The number of square pixels a blob must be before we consider it a
# candidate for tracking.
BLOB_SIZE = 500
# The number of pixels wide a blob must be before we consider it a
# candidate for tracking.
BLOB_WIDTH = 32
# The weighting to apply to "this" frame when averaging. A higher number
# here means that the average scene will pick up changes more readily,
# thus making the difference between average and current scenes smaller.
DEFAULT_AVERAGE_WEIGHT = 0.02
# The maximum distance a blob centroid is allowed to move in order to
# consider it a match to a previous scene's blob.
BLOB_LOCKON_DISTANCE_PX = 100
# The number of seconds a blob is allowed to sit around without having
# any new blobs matching it.
BLOB_TRACK_TIMEOUT = 0.7
# Blob smoothing function, to join 'gaps' in cars
SMOOTH = 8
# The left and right Y positions of the "poles". These are used to
# track the speed of a vehicle across the scene.
LEFT_POLE_PY = 190
RIGHT_POLE_PY = 190
# Constants for drawing on the frame.
LINE_THICKNESS = 1
CIRCLE_SIZE = 5
RESIZE_RATIO = 0.4

counter = 0

from itertools import tee
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

cap = cv2.VideoCapture('/Users/datascience9/Veh Detection/TFL API/625_201707211410.mp4')

# get frame size
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/Users/datascience9/Veh Detection/Outputs/625_201707211410_output.mp4', fourcc, 20, (w, h))
outblob = cv2.VideoWriter('/Users/datascience9/Veh Detection/Outputs/625_201707211410_outblob.mp4', fourcc, 20, (w, h))

# create a mask
mask = np.zeros((h,w), np.uint8)
mask[188:258, 0:210] = 255

# A variable to store the running average.
avg = None
# A list of "tracked blobs".
tracked_blobs = []

while(1):
    ret, frame = cap.read()
    
    if ret == True:
        # get returned time
        frame_time = time.time()
        
        # convert BGR to HSV
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # only use the Value channel of the frame
        (_,_,grayFrame) = cv2.split(frame)
        grayFrame = cv2.GaussianBlur(grayFrame, (21, 21), 0)
        
        # add mask to each frame
        #grayFrame = cv2.bitwise_and(grayFrame, grayFrame, mask=mask)
        
        if avg is None:
            # Set up the average if this is the first time through.
            avg = grayFrame.copy().astype("float")
            continue
        
        # Build the average scene image by accumulating this frame
        # with the existing average.
        cv2.accumulateWeighted(grayFrame, avg, DEFAULT_AVERAGE_WEIGHT)
        cv2.imshow("gray_average", cv2.convertScaleAbs(avg))
        
        # Compute the grayscale difference between the current grayscale frame and
        # the average of the scene.
        differenceFrame = cv2.absdiff(grayFrame, cv2.convertScaleAbs(avg))
        cv2.imshow("difference", differenceFrame)
        
        # Apply a threshold to the difference: any pixel value above the sensitivity
        # value will be set to 255 and any pixel value below will be set to 0.
        retval, thresholdImage = cv2.threshold(differenceFrame, THRESHOLD_SENSITIVITY, 255, cv2.THRESH_BINARY)
        
        # We'll need to fill in the gaps to make a complete vehicle as windows
        # and other features can split them!
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SMOOTH, SMOOTH))
        # Fill any small holes
        closing = cv2.morphologyEx(thresholdImage, cv2.MORPH_CLOSE, kernel)
        # Remove noise
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

        # Dilate to merge adjacent blobs
        thresholdImage = cv2.dilate(opening, kernel, iterations = 2)
        #thresholdImage = cv2.dilate(thresholdImage, None, iterations=2)
        
        cv2.imshow("threshold", thresholdImage)
        threshout = cv2.cvtColor(thresholdImage, cv2.COLOR_GRAY2BGR)
        outblob.write(threshout)

        # Find contours aka blobs in the threshold image.
        _, contours, hierarchy = cv2.findContours(thresholdImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        # Filter out the blobs that are too small to be considered cars.
        blobs = filter(lambda c: cv2.contourArea(c) > BLOB_SIZE, contours)
        
        if blobs:
            for c in blobs:
                # Find the bounding rectangle and center for each blob
                (x, y, w, h) = cv2.boundingRect(c)
                center = (int(x + w/2), int(y + h/2))
    
                # Draw the rectangle around the blob on the frame that we'll show in a UI later
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), LINE_THICKNESS)
                
                # counter algo
                dy = center[1]-LEFT_POLE_PY
                if -1 <= dy <= 1:
                    if 20 <= center[0] <= 165:
                        counter += 1
                        print(counter)
                
                # Look for existing blobs that match this one
                closest_blob = None
                if tracked_blobs:
                    # Sort the blobs we have seen in previous frames by pixel distance from this one
                    closest_blobs = sorted(tracked_blobs, key=lambda b: cv2.norm(b['trail'][0], center))
    
                    # Starting from the closest blob, make sure the blob in question is in the expected direction
                    for close_blob in closest_blobs:
                        distance = cv2.norm(center, close_blob['trail'][0])
    
                        # Check if the distance is close enough to "lock on"
                        if distance < BLOB_LOCKON_DISTANCE_PX:
                            # If it's close enough, make sure the blob was moving in the expected direction
                            expected_dir = close_blob['dir']
                            if expected_dir == 'left' and close_blob['trail'][0][0] < center[0]:
                                continue
                            elif expected_dir == 'right' and close_blob['trail'][0][0] > center[0]:
                                continue
                            else:
                                closest_blob = close_blob
                                break
    
                    if closest_blob:
                        # If we found a blob to attach this blob to, we should
                        # do some math to help us with speed detection
                        prev_center = closest_blob['trail'][0]
                        if center[0] < prev_center[0]:
                            # It's moving left
                            closest_blob['dir'] = 'left'
                            closest_blob['bumper_x'] = x
                        else:
                            # It's moving right
                            closest_blob['dir'] = 'right'
                            closest_blob['bumper_x'] = x + w
    
                        # ...and we should add this centroid to the trail of
                        # points that make up this blob's history.
                        closest_blob['trail'].insert(0, center)
                        closest_blob['last_seen'] = frame_time

                if not closest_blob:
                    # If we didn't find a blob, let's make a new one and add it to the list
                    b = dict(
                        id=str(uuid.uuid4())[:8],
                        first_seen=frame_time,
                        last_seen=frame_time,
                        dir=None,
                        bumper_x=None,
                        trail=[center],
                    )
                    tracked_blobs.append(b)
    
        if tracked_blobs:
            # Prune out the blobs that haven't been seen in some amount of time
            for i in range(len(tracked_blobs) - 1, -1, -1):
                if frame_time - tracked_blobs[i]['last_seen'] > BLOB_TRACK_TIMEOUT:
                    print("Removing expired track {}".format(tracked_blobs[i]['id']))
                    del tracked_blobs[i]
    

    
        # Show the image from the camera (along with all the lines and annotations)
        # in a window on the user's screen.
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        cv2.imshow("preview", frame)
        out.write(frame)

        if cv2.waitKey(27) & 0xFF == ord('q'):
                break
    else:
        break


cv2.destroyAllWindows()
cap.release()
out.release()
