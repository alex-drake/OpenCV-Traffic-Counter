#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:14:36 2017

@author: alexdrake
"""

import cv2
import numpy as np
import os

# get working directory
loc = os.path.abspath('')

# import image for detecting lane edges
img = cv2.imread(loc+"/trafficCounter/backgrounds/625_bg.jpg",0)

kernel = np.ones((5,5),np.uint8)
#Removing noise from image
blur = cv2.bilateralFilter(img, 11, 3, 3)
edges = cv2.Canny(img, 0, 820)
edges2 = cv2.Canny(img, 0, 800)

# get difference between edges
diff = cv2.absdiff(edges, cv2.convertScaleAbs(edges2))

laplacian = cv2.Laplacian(diff, cv2.CV_8UC1)

# Do a dilation and erosion to accentuate the triangle shape
dilated = cv2.dilate(laplacian, kernel, iterations = 2)
erosion = cv2.erode(dilated,kernel,iterations = 3)

# show erosion output to user
cv2.imshow("ero", erosion)
cv2.waitKey(0)

# find contours
im2, contours, hierarchy =  cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#keep 10 largest contours
cnts = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None

for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05 * peri, True)
    # if our approximated contour has three points, then
    # it must be the road markings
    if len(approx) == 4:
        screenCnt = approx
        break
cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)
cv2.imshow("Road markings", img)
cv2.waitKey(0)