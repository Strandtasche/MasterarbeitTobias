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

### convertPNGtoRGB.sh

_Purpose_:   
Bashscript um die gesammelten Daten vom Schüttgutsortierer, die zuvor nach PNG konvertiert wurden zu debayern
Ruft *debayer.py* für alle Daten mit den richtigen Parametern auf

_Usage_:  
* ggf. targetPrefix im File ändern
* cd [Ordner in denen die Subordner mit den Daten liegen]
* bash ~/MasterarbeitTobias/scripts/convertPNGtoRGB.sh

_Input_:  
    Ordnerstruktur mit BMP Bildern

_Output_:  
selbe Ordnerstruktur in einem Parallelordner mit allen Bildern als PNG

### convertRGBtoCSV.sh

_Purpose_:   
Bashscript um die gesammelten Daten vom Schüttgutsortierer, die zuvor debayered wurden, zu segmentieren
Ruft *segment.py* für alle Daten mit den richtigen Parametern auf (einzelne Bilder zu Dataset: "%05d" )

_Usage_:  
* ggf. targetPrefix im File ändern
* cd [Ordner in denen die Subordner mit den Daten liegen]
* bash ~/MasterarbeitTobias/scripts/convertPNGtoRGB.sh

_Input_:  
    Ordnerstruktur mit png Bildern aus convertBMPtoPNG

_Output_:  
selbe Ordnerstruktur in einem Parallelordner mit allen Bildern als debayertes PNG


### debayer.py

_Purpose_:  

[//]: <> converting the image given by the Schüttgutsortierer's camera from greyscale (bayered) to color. Demosaic frames from Bonito Camera.

Umrechnen ("Demosaicing") eines Bildes, das in Graustufen von der Kamera des Schüttgutsortierers kommt, in RGB.

_Usage_:  
* -i: input file or folder - required
* -o: output file or folder - required
* -e: file extension of input files - optional. Default is png

_Input_:  
bayered image(s) from the bonito camera.

_Output_:  
rgb image(s)

### renameScript.sh

_Purpose_:
renaming the "data.csv" files in different subfolders of the svn into more sensible names based on the folders they are in
(One off script, has to be updated for other circumstances)

_Input_:
No input, just 'bash renameScript.sh'

_Output_:
renamed files in svn (needs to be committed though)

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

### segment.py und segment_paramters.py


_Purpose_:  
converting the image given by the Schüttgutsortierer's camera from greyscale (bayered) to color.
Demosaic frames from Bonito Camera.

_Usage_:  
* -i: input dataset - required 
* -o: output file or folder - required
* -nd: No Display - optional. Schaltet Konsolenoutput aus - Beschleunigung bei großen Anzahlen von Bildern

_Input_:  
Dataset: Dateipfad zu Bildern mit Nummern verallgemeintert als "%05d" - siehe *convertRGBtoCSV.sh* script

_Output_:  
CSV Datei mit einer Zeile pro Bild: FrameNr, Anzahl detektierter Partikel und dann einer Liste mit den X- und Y-Koordinaten der detektierten Partikelmittelpunkte

### extract_blobs.py

**DEPRECATED!**
use segment.py indead

Written by Tobias Kronauer, minor edits by me.

_Purpose_:
This script serves to read images given the  provided input-directory (if none provided,
it uses the current directory) and given the file-extension of the images. After reading the images
it subtracts the background and calculates the blobs (= particles). Those are extracted afterwards
and saved in the provided output-directory.

Additionally, important data is saved into a csv-file.

_Input_:
* input_directory: the directory which contains the images (if none: current_directory); path is relative
* file_extension: file-extension of the images which shall be looked for, other files are ignored
* output_directory: directory the extracted blobs shall be saved to

_Output_:
Blobs in folders and csv-file

### extract_blobs_iterator.sh

**DEPRECATED!**
Benutzen um das *extract_blobs.py* script auf allen aufgenommenen Batches auszuführen

### doAllThingsMatlab.m

**DEPRECATED!**
Stattdessen *create_track_history.m* benutzen

_Purpose_:   
Anwenden des TrackSort Algorithmus auf die rohen Daten.

_Usage_: In Matlab  
 1. add nessesary things to path - _addpath(genpath('matlab') ?_
 2. navigate to folder with data you want to process
 3. optional: change files variable to fit your input csvs.
 4. \>\> doAllThingsMatlab

