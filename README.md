# MasterarbeitTobias
Masterarbeit Sommersemester 2018, am KIT - ISAS, Karlsruhe


## Scripts

An dieser Stelle sollen die Verschiedenen Scripts die im Repo zu finden sind mit einer kurzen Zusammenfassung aufgezählt werden.

### reorderCentroids.py

Purpose: Umsortieren von rohen Daten, sodass eine Spalte einem Partikel zugeordnet ist

USAGE: python reorderCentroids.py inputFile, outputFile, [track histories]

Input: Unsortiertes CSV mit X, Y positionen von Partikeln zu einzelnen Zeitschritten, TrackHistory MeasuredIndex Listen

Output: CSV, wobei eine Spalte die Bewegung eines einzelnen Teilchens abbildet.

### doAllThingsMatlab.m

Purpose: Anwenden des TrackSort Algorithmus auf die rohen Daten.

USAGE: In Matlab
 1. add nessesary things to path - _addpath(genpath('matlab') ?_
 2. navigate to folder with data you want to process
 3. optional: change files variable to fit your input csvs.
 4. >> doAllThingsMatlab

