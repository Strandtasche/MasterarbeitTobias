#!/usr/bin/env python

import os
import csv
from numpy import genfromtxt
import sys


cwd = os.getcwd()
# print(cwd)

#USAGE ARGUMENTS: input, output, [track histories]

inputFile = cwd + "/../data/test_case1.csv"
trackHistory  = []
trackHistory.append(cwd + "/../data/1output.csv")
trackHistory.append(cwd + "/../data/2output.csv")
trackHistory.append(cwd + "/../data/3output.csv")

outputFile = cwd + "/../data/test_case1_ordered.csv"

if (len(sys.argv) > 1):
        inputFile = os.path.abspath(sys.argv[1])
        outputFile = os.path.abspath(sys.argv[2])
        trackHistory = []
        for i in range (3, len(sys.argv)):
            trackHistory.append(os.path.abspath(sys.argv[i]))

my_data = []
for track in trackHistory:
    my_data.append(genfromtxt(track, delimiter=',', dtype='int_'))

print("Number tracks: " + str(len(trackHistory)))

# print(my_data[0])
dataTotal = []

with open(inputFile, "r") as f:
    reader = csv.reader(f)
    header = next(reader, None) #skip header line
    i = 0
    for row in reader:
        data = []
        data.append(row[0])
        data.append(row[1])
        for track in my_data:
            # print(track)
            data.append(row[2+2*(track[i]-1)]) #X index conversion from matlab
            data.append(row[2+2*(track[i]-1)+1]) #Y index conversion from matlab
        i = i + 1
        dataTotal.append(data)

print("successfully read from " + inputFile)

#print(header[0])
#print(dataTotal)##

with open(outputFile, "w") as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(header)
    for line in dataTotal:
        writer.writerow(line)


print("exported to " + outputFile)

