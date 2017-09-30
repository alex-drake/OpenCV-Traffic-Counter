#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 10:28:47 2017

@author: alexdrake
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# get working directory
loc = os.path.abspath('')

cap = cv2.VideoCapture(loc+'/trafficCounter/inputs/625_201709301058.mp4')

# get frame size
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# create a mask
mask1 = np.zeros((h,w), np.uint8)
mask1[188:258, 30:140] = 255

mask2 = np.zeros((h,w), np.uint8)
mask2[188:258, 190:260] = 255

while(1):
    ret, frame = cap.read()
    
    if ret == True:
        # convert BGR to HSV
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # add mask to each frame
        frame1 = cv2.bitwise_and(frame, frame, mask=mask1)
        frame2 = cv2.bitwise_and(frame, frame, mask=mask2)
        
        #hist_full = cv2.calcHist([frame],[2],None,[256],[0, 256])
        hist_mask1 = cv2.calcHist([frame1],[2],mask1,[256],[50, 256])
        hist_mask2 = cv2.calcHist([frame2],[2],mask2,[256],[50, 256])
        
        # calculate summary stats
        stats = np.array([["Count",np.count_nonzero(hist_mask1)],
                           ["Max",np.max(hist_mask1)],
                           ["Mean",np.mean(hist_mask1)],
                           ["Std",np.std(hist_mask1)],
                           ["25%",np.percentile(hist_mask1,25)],
                           ["50%",np.percentile(hist_mask1,50)],
                           ["75%",np.percentile(hist_mask1,75)]])
        print(stats)
        
        # create plots
        plt.subplot(221), plt.imshow(frame1, 'gray')
        plt.subplot(222), plt.plot(hist_mask1)
        plt.xlim([0,256]), plt.ylim([0,300])
        plt.subplot(223), plt.imshow(frame2, 'gray')
        plt.subplot(224), plt.plot(hist_mask2)
        plt.xlim([0,256]), plt.ylim([0,300]) 
        
        # plot histogram  
        plt.show()
        
        #out.write()
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        break


cv2.destroyAllWindows()
cap.release()
out.release()
