#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 10:29:21 2017

@author: datascience9
"""

import os
import glob

# get working directory
loc = os.path.abspath('')

# import dataset containing cars and non-cars
cars = glob.glob("/Users/datascience9/Veh Detection/Udacity Data/vehicles/*/*.png")
non_cars = glob.glob("/Users/datascience9/Veh Detection/Udacity Data/non-vehicles/*/*.png")

def create_pos_n_neg():
    for file_type in ['non_cars']:
        
        for img in os.listdir('non-vehicles/GTI/'):
            if file_type == 'non_cars':
                line = 'non-vehicles/GTI/'+img+'\n'
                with open('bg.txt','a') as f:
                    f.write(line)
        for img in os.listdir('non-vehicles/Extras/'):
            if file_type == 'non_cars':
                line = 'non-vehicles/Extras/'+img+'\n'
                with open('bg.txt','a') as f:
                    f.write(line)
                    
    for file_type in ['cars']:
        
        for img in os.listdir('vehicles/GTI_Far/'):
            if file_type == 'cars':
                line = 'vehicles/GTI_Far/'+img+' 1 0 0 64 64\n'
                with open('info.lst','a') as f:
                    f.write(line)
        for img in os.listdir('vehicles/GTI_Left/'):
            if file_type == 'cars':
                line = 'vehicles/GTI_Left/'+img+' 1 0 0 64 64\n'
                with open('info.lst','a') as f:
                    f.write(line)
        for img in os.listdir('vehicles/GTI_MiddleClose/'):
            if file_type == 'cars':
                line = 'vehicles/GTI_MiddleClose/'+img+' 1 0 0 64 64\n'
                with open('info.lst','a') as f:
                    f.write(line)
        for img in os.listdir('vehicles/GTI_Right/'):
            if file_type == 'cars':
                line = 'vehicles/GTI_Right/'+img+' 1 0 0 64 64\n'
                with open('info.lst','a') as f:
                    f.write(line)
        for img in os.listdir('vehicles/KITTI_extracted/'):
            if file_type == 'cars':
                line = 'vehicles/KITTI_extracted/'+img+' 1 0 0 64 64\n'
                with open('info.lst','a') as f:
                    f.write(line)
                    
create_pos_n_neg()

