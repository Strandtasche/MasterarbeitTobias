# MasterarbeitTobias
Masterarbeit Sommersemester 2018, am KIT - ISAS, Karlsruhe


## Scripts

An dieser Stelle sollen die Verschiedenen Scripts die im Repo zu finden sind mit einer kurzen Zusammenfassung aufgezählt werden.

### convertBMPtoPNG.sh

_Purpose_:   
Bashscript um die gesammelten Daten vom Schüttgutsortierer von BMP in PNG zu konvertieren

_Usage_:  
* ggf. targetPrefix im File ändern
* cd [Ordner in denen die Subordner mit den Daten liegen]
* bash ~/MasterarbeitTobias/scripts/convertBMPtoPNG.sh

_Input_:  
    Ordnerstruktur mit BMP Bildern

_Output_:  
selbe Ordnerstruktur in einem Parallelordner mit allen Bildern als PNG


### debayer.py

_Purpose_:  
converting the image given by the Schüttgutsortierer's camera from greyscale (bayered) to color.
Demosaic frames from Bonito Camera.

_Usage_:  
* -i: input file or folder - required
* -o: output file or folder - required
* -e: file extension of input files - optional. Default is png

_Input_:  
bayered image(s) from the bonito camera.

_Output_:  
rgb image(s)


### exportTensorFlowLog.py

_Purpose_:  
takes event.out.tfevents file, extracts the scalars and writes them to a csv file.

_Usage_:  
python exportTensorFlowLog.py "inputFile" "outputFolder"

_Input_:  
[todo]


### reorderCentroids.py

_Purpose_:   
Umsortieren von rohen Daten, sodass eine Spalte einem Partikel zugeordnet ist

_Usage_:   
python reorderCentroids.py inputFile, outputFile, [track histories]

_Input_:   
Unsortiertes CSV mit X, Y positionen von Partikeln zu einzelnen Zeitschritten, TrackHistory MeasuredIndex Listen

_Output_:  
CSV, wobei eine Spalte die Bewegung eines einzelnen Teilchens abbildet.

### doAllThingsMatlab.m

**DEPRECATED!**
use create_track_history.m instead

_Purpose_:   
Anwenden des TrackSort Algorithmus auf die rohen Daten.

_Usage_: In Matlab  
 1. add nessesary things to path - _addpath(genpath('matlab') ?_
 2. navigate to folder with data you want to process
 3. optional: change files variable to fit your input csvs.
 4. \>\> doAllThingsMatlab

