#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 12:59:22 2017

@author: alexdrake
"""
import numpy as np
import cv2
import os

# get working directory
loc = os.path.abspath('')

#this is the cascade
car_cascade = cv2.CascadeClassifier(loc+'/trafficCounter/classifier/cars3.xml')

# Video source
inputFile = loc+'/trafficCounter/inputs/625_201709141311.mp4'
cap = cv2.VideoCapture(inputFile)

# get frame size
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# create a mask (manual for each camera)
mask = np.zeros((frame_h,frame_w), np.uint8)
mask[:,:] = 255
mask[:120, :] = 0

frame_no = 0

ret, img = cap.read()

while ret:    
    ret, img = cap.read()
    frame_no = frame_no + 1

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # apply mask
    gray = cv2.bitwise_and(gray, gray, mask = mask)

    # image, reject levels level weights.
    cars = car_cascade.detectMultiScale(gray, 1.008, 5)
    
    # add this
    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
    print('Processing %d : cars detected : [%s]' % (frame_no, len(cars)))
    
    cv2.imshow('img',img)
    if cv2.waitKey(27) and 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()