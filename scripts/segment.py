# -*- coding: utf-8 -*-
"""
Created on Fri May  4 12:26:24 2018

@author: ma
"""

import numpy as np
import cv2
import argparse
from segment_parameters import SegmentParameters
import csv

WIDTH = 2320
HEIGHT = 1700

FLOATPRECISION = 4

font                   = cv2.FONT_HERSHEY_SIMPLEX
lineType               = 2

def get_centroid(cnt):
    M = cv2.moments(cnt)
    
    cx = (M['m10']/M['m00'])
    cy = (M['m01']/M['m00'])
    
    return np.array([cx, cy])

def is_valid_rect(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    
    if x == 0:
        return False
    if x + w >= WIDTH - 1:
        return False
    if y == 0:
        return False
    if y + h >= HEIGHT - 1:
        return False
    
    return True

def segment(dataset, display):
    cap = cv2.VideoCapture(dataset)
    segment_parameters = SegmentParameters()
    paras = segment_parameters.get_parameters(dataset)
    all_centroids = []
    
    counter = 0
    
    while(1):
        ret, frame = cap.read()
        
        if ret == False:
            break
        
        
        # cv2.imshow('image', frame)
        
        # cv2.waitKey(0)
        
        frame_centroids = []
        
        frame = frame[:HEIGHT,:WIDTH,:]
        
        frame_cvt = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        
        h, s, b = cv2.split(frame_cvt)
        
        # print([h[i][40:45] for i in range(1570,1575)])
        
        h = cv2.inRange(h, paras.h_lower, paras.h_upper)
        s = cv2.inRange(s, paras.s_lower, paras.s_upper)
        b = cv2.inRange(b, paras.b_lower, paras.b_upper)
        
        # print("H: {}, S: {}, B: {}".format(len(h), len(s), len(b)))
        # print(h)
        
        seg = cv2.bitwise_and(h, s)
        seg = cv2.bitwise_and(seg, b)
        
        # print(seg[1100:1140])
        
        # OPTIONAL - KANN BEI TRENNUNG VON SICH BERÜHRENDEN OBJEKTEN HELFEN!
        #        seg = cv2.erode(seg, kernel, iterations=3)
        #        seg = cv2.dilate(seg, kernel, iterations=3)
        
        _, contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hulls = [c for c in contours if cv2.contourArea(c) > paras.min_area]
        hulls = [h for h in hulls if is_valid_rect(h)]
        
        # print("Hulls size: {}".format([cv2.contourArea(c) for c in hulls]))
        
        centroids = [get_centroid(h) for h in hulls]
        
        [frame_centroids.append(c) for c in centroids]
        all_centroids.append(np.array(frame_centroids))
        
        if display:
            cv2.drawContours(frame, hulls, -1, (0, 0, 255), 3)
            show_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            
            cv2.imshow('frame', show_frame)
            
            k = cv2.waitKey(50) & 0xff
            if k == 27:
                break
        print(counter)
        counter = counter + 1
    
    np.save(dataset, np.array(all_centroids))
    # print(all_centroids)
    cap.release()
    cv2.destroyAllWindows()
    
    return all_centroids


def writeToCSV(data, target):
    with open(target + '.csv', 'w') as csvfile:
        counter = 0
        writer = csv.writer(csvfile, delimiter=',')
        formatString = '{:.' + str(FLOATPRECISION) + 'f}'
        maxDetecNum = len(max(data, key=len))
        columnCount = 2 + 2 * maxDetecNum
        header = ['FrameNr', 'NumberMidPoints']
        for i in range(1, maxDetecNum + 1):
            header.append('MidPoint_{}_x'.format(i))
            header.append('MidPoint_{}_y'.format(i))
        
        writer.writerow(header)
        
        for elem in data:
            row = [counter, len(elem)]
            for k in elem:
                row.append(k[0])
                row.append(k[1])
            precRow = [formatString.format(tid) for tid in row]
            if len(precRow) < columnCount:
                precRow = precRow + (['NaN'] * (columnCount - len(row)))
            writer.writerow(precRow)
            counter = counter + 1
        

def main():
    print('Parsing command line...')
    parser = argparse.ArgumentParser(description='Segment tracks.')
    parser.add_argument('-i', type=str, help='dataset', required=True)
    parser.add_argument('-nd', action="store_true", help='no display')
    parser.add_argument('-o', type=str, help='output Target', required=True)
	
    
    args = parser.parse_args()
    display = True
    if args.nd == True:
        display = False
    
    print('Processing...')
    
    data = segment(args.i, display)
    writeToCSV(data, args.o)
    
    print('Done.')

if __name__ == "__main__":
    main()

