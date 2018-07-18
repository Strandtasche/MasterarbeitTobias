# MasterarbeitTobias
Masterarbeit Sommersemester 2018, am KIT - ISAS, Karlsruhe


## Scripts

An dieser Stelle sollen die Verschiedenen Scripts die im Repo zu finden sind mit einer kurzen Zusammenfassung aufgezählt werden.

### convertBMPtoPNG.sh

Purpose: Bashscript um die gesammelten Daten vom Schüttgutsortierer von BMP in PNG zu konvertieren

Usage:
ggf. targetPrefix im File ändern
cd [Ordner in denen die Subordner mit den Daten liegen]
bash ~/MasterarbeitTobias/scripts/convertBMPtoPNG.sh

Input: Ordnerstruktur mit BMP Bildern

Output: selbe Ordnerstruktur in einem Parallelordner mit allen Bildern als PNG


### reorderCentroids.py

Purpose: Umsortieren von rohen Daten, sodass eine Spalte einem Partikel zugeordnet ist

USAGE: python reorderCentroids.py inputFile, outputFile, [track histories]

Input: Unsortiertes CSV mit X, Y positionen von Partikeln zu einzelnen Zeitschritten, TrackHistory MeasuredIndex Listen

Output: CSV, wobei eine Spalte die Bewegung eines einzelnen Teilchens abbildet.

### doAllThingsMatlab.m

**DEPRECATED!**
use create_track_history.m instead

Purpose: Anwenden des TrackSort Algorithmus auf die rohen Daten.

USAGE: In Matlab
 1. add nessesary things to path - _addpath(genpath('matlab') ?_
 2. navigate to folder with data you want to process
 3. optional: change files variable to fit your input csvs.
 4. \>\> doAllThingsMatlab

