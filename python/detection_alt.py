#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 10:28:47 2017

@author: datascience9
"""

import cv2
import numpy as np
import time
import logging
import math
import csv
import re

# Vehicle_counter from Dan Maesks response on 
# https://stackoverflow.com/questions/36254452/counting-cars-opencv-python-issue/36274515#36274515

# for testing
tracked_blobs = []
tracked_conts = []
t_retval = []

# ============================================================================

class Vehicle(object):
    def __init__(self, id, position):
        self.id = id
        self.positions = [position]
        self.frames_since_seen = 0
        self.frames_seen = 0
        self.counted = False

    @property
    def last_position(self):
        return self.positions[-1]

    def add_position(self, new_position):
        self.positions.append(new_position)
        self.frames_since_seen = 0
        self.frames_seen += 1

    def draw(self, output_image):
        for point in self.positions:
            cv2.circle(output_image, point, 2, (0, 0, 255), -1)
            cv2.polylines(output_image, [np.int32(self.positions)]
                , False, (0, 0, 255), 1)

# ============================================================================

class VehicleCounter(object):
    def __init__(self, shape, divider):
        self.log = logging.getLogger("vehicle_counter")

        self.height, self.width = shape
        self.divider = divider

        self.vehicles = []
        self.next_vehicle_id = 0
        self.vehicle_count = 0
        self.max_unseen_frames = 7


    @staticmethod
    def get_vector(a, b):
        """Calculate vector (distance, angle in degrees) from point a to point b.

        Angle ranges from -180 to 180 degrees.
        Vector with angle 0 points straight down on the image.
        Values increase in clockwise direction.
        """
        dx = float(b[0] - a[0])
        dy = float(b[1] - a[1])

        distance = math.sqrt(dx**2 + dy**2)

        if dy > 0:
            angle = math.degrees(math.atan(-dx/dy))
        elif dy == 0:
            if dx < 0:
                angle = 90.0
            elif dx > 0:
                angle = -90.0
            else:
                angle = 0.0
        else:
            if dx < 0:
                angle = 180 - math.degrees(math.atan(dx/dy))
            elif dx > 0:
                angle = -180 - math.degrees(math.atan(dx/dy))
            else:
                angle = 180.0        

        return distance, angle 


    @staticmethod
    def is_valid_vector(a):
        distance, angle = a
        threshold_distance = 12.0
        #threshold_distance = max(10.0, -0.008 * angle**2 + 0.4 * angle + 25.0)
#        if angle > 0:
#            threshold_distance = 0
#        else:
#            threshold_distance = max(8.0, -0.001*(angle + 90)**2 + 12.0)

        return (distance <= threshold_distance)


    def update_vehicle(self, vehicle, matches):
        # Find if any of the matches fits this vehicle
        for i, match in enumerate(matches):
            contour, centroid = match
            
            # store the vehicle data
            vector = self.get_vector(vehicle.last_position, centroid)
            b = dict(
                    id = vehicle.id,
                    center_x = centroid[0],
                    center_y = centroid[1],
                    vector_x = vector[0],
                    vector_y = vector[1],
                    counted = vehicle.counted,
                    frame_number = frame_no
                    )
            tracked_blobs.append(b)
            
            if self.is_valid_vector(vector):
                vehicle.add_position(centroid)
                vehicle.frames_seen += 1
                self.log.debug("Added match (%d, %d) to vehicle #%d. vector=(%0.2f,%0.2f)"
                    , centroid[0], centroid[1], vehicle.id, vector[0], vector[1])
                return i

        # No matches fit...        
        vehicle.frames_since_seen += 1
        self.log.debug("No match for vehicle #%d. frames_since_seen=%d"
            , vehicle.id, vehicle.frames_since_seen)

        return None


    def update_count(self, matches, output_image = None):
        self.log.debug("Updating count using %d matches...", len(matches))

        # First update all the existing vehicles
        for vehicle in self.vehicles:
            i = self.update_vehicle(vehicle, matches)
            if i is not None:
                del matches[i]

        # Add new vehicles based on the remaining matches
        for match in matches:
            contour, centroid = match
            new_vehicle = Vehicle(self.next_vehicle_id, centroid)
            self.next_vehicle_id += 1
            self.vehicles.append(new_vehicle)
            self.log.debug("Created new vehicle #%d from match (%d, %d)."
                , new_vehicle.id, centroid[0], centroid[1])

        # Count any uncounted vehicles that are past the divider
        for vehicle in self.vehicles:
            if not vehicle.counted and (vehicle.last_position[1] > self.divider) and (vehicle.frames_seen > 2):
            #if not vehicle.counted and (vehicle.frames_seen > 2):    
                self.vehicle_count += 1
                vehicle.counted = True
                self.log.debug("Counted vehicle #%d (total count=%d)."
                    , vehicle.id, self.vehicle_count)

        # Optionally draw the vehicles on an image
        if output_image is not None:
            for vehicle in self.vehicles:
                vehicle.draw(output_image)

            cv2.putText(output_image, ("%02d" % self.vehicle_count), (142, 10)
                , cv2.FONT_HERSHEY_PLAIN, 1, (127, 255, 255), 1)

        # Remove vehicles that have not been seen long enough
        removed = [ v.id for v in self.vehicles
            if v.frames_since_seen >= self.max_unseen_frames ]
        self.vehicles[:] = [ v for v in self.vehicles
            if not v.frames_since_seen >= self.max_unseen_frames ]
        for id in removed:
            self.log.debug("Removed vehicle #%d.", id)

        self.log.debug("Count updated, tracking %d vehicles.", len(self.vehicles))

# ============================================================================

# Video source
inputFile = '/Users/datascience9/Veh Detection/TFL API/625_201708101116.mp4'
#inputFile = '/Users/datascience9/Veh Detection/RNC Data/1425_Tottenham.mp4'
camera = re.match(r".*/(\d+)_.*", inputFile)
camera = camera.group(1)

# import video file
cap = cv2.VideoCapture(inputFile)

# Default background (as clear as possible or pre-created avg frame!!)
try:
    bg = "/Users/datascience9/Veh Detection/Sample Scripts/test_images/"+camera+"_bg.jpg"
    default_bg = cv2.imread(bg)
except TypeError:
    avg = None

if 'default_bg' in locals():
    default_bg = cv2.cvtColor(default_bg, cv2.COLOR_BGR2HSV)
    (_,avgSat,default_bg) = cv2.split(default_bg)
    avg = default_bg.copy().astype("float")
    #avgSat = avgSat.copy().astype("float")
else:
    avg = None

  
# get frame size
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# The cutoff for threshold. A lower number means smaller changes between
# the average and current scene are more readily detected.
THRESHOLD_SENSITIVITY = 20
t_retval.append(THRESHOLD_SENSITIVITY)
# Blob size limit before we consider it for tracking.
CONTOUR_WIDTH = 21
CONTOUR_HEIGHT = 21
# The weighting to apply to "this" frame when averaging. A higher number
# here means that the average scene will pick up changes more readily,
# thus making the difference between average and current scenes smaller.
DEFAULT_AVERAGE_WEIGHT = 0.01
INITIAL_AVERAGE_WEIGHT = DEFAULT_AVERAGE_WEIGHT / 50
# The number of seconds a blob is allowed to sit around without having
# any new blobs matching it.
BLOB_TRACK_TIMEOUT = 5#1.2 #0.7
# Blob smoothing function, to join 'gaps' in cars
SMOOTH = max(2,int(round((CONTOUR_WIDTH**0.5)/2,0)))
# Constants for drawing on the frame.
LINE_THICKNESS = 1

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = '/Users/datascience9/Veh Detection/Outputs/'+camera+'_output.mp4'
outblob = '/Users/datascience9/Veh Detection/Outputs/'+camera+'_outblob.mp4'
diffop = '/Users/datascience9/Veh Detection/Outputs/'+camera+'_outdiff.mp4'
out = cv2.VideoWriter(out, fourcc, 20, (frame_w, frame_h))
outblob = cv2.VideoWriter(outblob, fourcc, 20, (frame_w, frame_h))
diffop = cv2.VideoWriter(diffop, fourcc, 20, (frame_w, frame_h))

# A list of "tracked blobs".
blobs = []
car_counter = None  # will be created later
frame_no = 0

while(1):
    ret, frame = cap.read()
    
    if ret == True:
        frame_no = frame_no + 1
        
        print("Processing frame ",frame_no)
        
        # get returned time
        frame_time = time.time()
        
        # convert BGR to HSV
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # only use the Value channel of the frame
        (_,satFrame,grayFrame) = cv2.split(frame)
        #grayFrame = cv2.GaussianBlur(grayFrame, (21, 21), 0)
        grayFrame = cv2.bilateralFilter(grayFrame, 11, 21, 21)
        #satFrame = cv2.bilateralFilter(satFrame, 11, 21, 21)

        if avg is None:
            # Set up the average if this is the first time through.
            avg = grayFrame.copy().astype("float")
            #avgSat = satFrame.copy().astype("float")
            continue
        
        # Build the average scene image by accumulating this frame
        # with the existing average.
        if frame_no < 10:
            def_wt = INITIAL_AVERAGE_WEIGHT
        else:
            def_wt = DEFAULT_AVERAGE_WEIGHT
            
        cv2.accumulateWeighted(grayFrame, avg, def_wt)
        #cv2.imshow("gray_average", cv2.convertScaleAbs(avg))
        #cv2.accumulateWeighted(satFrame, avgSat, def_wt)
        
        # export averaged background for use in next video feed run
        if frame_no > 250:
            grayOp = cv2.cvtColor(cv2.convertScaleAbs(avg), cv2.COLOR_GRAY2BGR)
            backOut = "/Users/datascience9/Veh Detection/Sample Scripts/test_images/"+camera+"_bg.jpg"
            cv2.imwrite(backOut, grayOp)
        
        # Compute the grayscale difference between the current grayscale frame and
        # the average of the scene.
        differenceFrame = cv2.absdiff(grayFrame, cv2.convertScaleAbs(avg))
        # blur the difference image
        differenceFrame = cv2.GaussianBlur(differenceFrame, (5, 5), 0)
        #differenceFrame = cv2.bilateralFilter(differenceFrame, 11, 33, 33)
        cv2.imshow("difference", differenceFrame)
        diffout = cv2.cvtColor(differenceFrame, cv2.COLOR_GRAY2BGR)
        diffop.write(diffout)
        
        edge = cv2.Canny(differenceFrame, 80, 200, 5)
        cv2.imshow("edge", edge)
#        satFrame = cv2.absdiff(satFrame, cv2.convertScaleAbs(avgSat))
#        satFrame = cv2.GaussianBlur(satFrame, (5, 5), 0)
#
#        _, satImg = cv2.threshold(satFrame, 0, 255,
#                                    cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#        cv2.imshow("satThresh",satImg)
        
        # Apply a threshold to the difference: any pixel value above the sensitivity
        # value will be set to 255 and any pixel value below will be set to 0.
#        retval, thresholdImage = cv2.threshold(differenceFrame, THRESHOLD_SENSITIVITY, 
#                                               255, cv2.THRESH_BINARY)
        # get estimated otsu threshold level
        retval, _ = cv2.threshold(differenceFrame, 0, 255,
                                  cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # add to list of threshold levels
        t_retval.append(retval)
        
        # apply threshold based on average threshold value so far
        ret2, thresholdImage = cv2.threshold(differenceFrame, 
                                             int(np.mean(t_retval)*0.9),
                                             255, cv2.THRESH_BINARY)
        
        # We'll need to fill in the gaps to make a complete vehicle as windows
        # and other features can split them!
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SMOOTH, SMOOTH))
        # Fill any small holes
        thresholdImage = cv2.morphologyEx(thresholdImage, cv2.MORPH_CLOSE, kernel)
        
        # Remove noise
        thresholdImage = cv2.morphologyEx(thresholdImage, cv2.MORPH_OPEN, kernel)

        # Dilate to merge adjacent blobs
        thresholdImage = cv2.dilate(thresholdImage, kernel, iterations = 2)
        
        cv2.imshow("threshold", thresholdImage)
        threshout = cv2.cvtColor(thresholdImage, cv2.COLOR_GRAY2BGR)
        outblob.write(threshout)

        # Find contours aka blobs in the threshold image.
        _, contours, hierarchy = cv2.findContours(thresholdImage, 
                                                  cv2.RETR_EXTERNAL, 
                                                  cv2.CHAIN_APPROX_SIMPLE)
        
        print("Found ",len(contours)," vehicle contours.")
        
        if contours:
            for (i, contour) in enumerate(contours):    
                # Find the bounding rectangle and center for each blob
                (x, y, w, h) = cv2.boundingRect(contour)
                contour_valid = (w > CONTOUR_WIDTH) and (h > CONTOUR_HEIGHT)
                
                print("Contour #",i,": pos=(x=",x,", y=",y,") size=(w=",w,
                      ", h=",h,") valid=",contour_valid)
                
                if not contour_valid:
                    continue
                
                center = (int(x + w/2), int(y + h/2))
                blobs.append(((x, y, w, h), center))
                    
        for (i, match) in enumerate(blobs):
            contour, centroid = match
            x, y, w, h = contour
            
            # store the contour data
            c = dict(
                        frame_no = frame_no,
                        centre_x = x,
                        centre_y = y,
                        width = w,
                        height = h
                        )
            tracked_conts.append(c)
            
            cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1), (0, 0, 255), LINE_THICKNESS)
            cv2.circle(frame, centroid, 2, (0, 0, 255), -1)
        
        if car_counter is None:
            print("Creating vehicle counter...")
            car_counter = VehicleCounter(frame.shape[:2], 2*frame.shape[0] / 3)
            
        
        # draw dividing line
        cv2.line(frame, (0, int(2*frame_h/3)),(frame_w, int(2*frame_h/3)),
                 (0,0,255), LINE_THICKNESS)

        car_counter.update_count(blobs, frame)
        
        # output video
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        cv2.imshow("preview", frame)
        out.write(frame)

        if cv2.waitKey(27) & 0xFF == ord('q'):
                break
    else:
        break

cv2.line()
cv2.destroyAllWindows()
cap.release()
out.release()
