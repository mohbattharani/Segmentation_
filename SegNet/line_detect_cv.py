#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 22:55:50 2017

@author: mohbat
"""

import cv2
import numpy as np
#import matplotlib.pyplot as plt
#from detect_lane_cv import draw_lines

def detect_two_lanes(lines):
    left_line = []
    right_line = []
    lenght_left = 0
    lenght_right = 0
    for line in lines:
        for x1,y1,x2,y2 in line:
            l = np.sqrt((y2-y1)**2 + (x2-x1)**2)
            m = (y2-y1)/(x2-x1)
            c = y1 - m*x1
            print ('m:',m,' c:',c,'  l:',l )

            if (m<0 and l>lenght_left):
                lenght_left = l
                left_line = line
            if (m>0 and l>lenght_right):
                lenght_right = l
                right_line = line

    return [left_line, right_line]

def remove_multiples(lines):
    new_lines = []
    slopes = []
    const = []
    k = 1
    need_append = True;
    for line in lines:
        for x1,y1,x2,y2 in line:
            l = np.sqrt((y2-y1)**2 + (x2-x1)**2)
            m = (y2-y1)/(x2-x1)
            c = y1 - m*x1
            #print ('m:',m,' c:',c,'  l:',l )
            if (k == 1):
                slopes.append(m)
                const.append(c)
                new_lines.append(line)
                k= k +1
            else:
                need_append = True
                for i,j in zip(slopes, const):
                    if (abs(m-i)<0.5 and abs(c-j)<50):
                        need_append = False
                if (need_append):
                    slopes.append(m)
                    const.append(c)
                    new_lines.append(line)


    return new_lines

def draw_hough_lines (image, lines):
    color = [255,0,0]
    thickness = 2
    i = 0
    lines = remove_multiples(lines)

    for line in lines:
        i = i+1
        for x1,y1,x2,y2 in line:
            m = (y2-y1)/(x2-x1)
            c = y1 - m*x1
            l = np.sqrt((y2-y1)**2 + (x2-x1)**2)
            #print ('sm:',m,' c:',c,'  l:',l )
            cv2.line(image, (x1,y1),(x2,y2), color, thickness)

    return image



def detect_lines (im):
    #im_hls = cv2.cvtColor (im, cv2.COLOR_RGB2HLS)
    #lower = np.uint8([0,200,0])  #filter white
    #upper = np.uint8([250,255,255])
    #white = cv2.inRange(im_hls, lower, upper)
    #lower = np.uint8([10,0,100])  # filter yellow
    #upper = np.uint8([40,255,255])
    #yellow = cv2.inRange(im_hls, lower, upper)
    #im_filtered = cv2.bitwise_or(white, yellow)
    #im_filtered = cv2.bitwise_and (im, im, mask = im_filtered)
    im_gray = cv2.cvtColor (im, cv2.COLOR_BGR2GRAY)
    im_smooth = cv2.GaussianBlur(im_gray, (5,5), 0)
    im_edges = cv2.Canny(im_smooth, 30, 130)
    lines = cv2.HoughLinesP(im_edges, rho=1, theta =np.pi/180, threshold =50, minLineLength=50, maxLineGap=100)

    return lines

#print ('start')
#im_filename = '/home/mohbat/RoadSegmentation/road lane Seg/SegNet11/road/road_rgb1.png'
#im = cv2.imread(im_filename)
#lines = detect_lines(im)
#im = draw_hough_lines(im, lines)
#cv2.imshow('Road Lanes', im)
#cv2.waitKey(0)
