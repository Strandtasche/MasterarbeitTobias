#!/usr/bin/env bash

echo "Starting Execution"
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

modelFolder="/home/hornberger/MasterarbeitTobias/models/final/simulated/separator-Final/simulated_Spheres"
python DNNRegressor-Example.py --training --custom --augment --filter --load /home/hornberger/MasterarbeitTobias/data/simulated/downsampled/h5/Spheres-Separator-FS7-Augm-Filtered.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/Separator-Final/filteredParticles-Sim-Spheres-final.json --overrideModel $modelFolder 2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Separator-Final/SimulatedSpheres_Final.txt
#(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV1"

modelFolder="/home/hornberger/MasterarbeitTobias/models/final/simulated/separator-Final/simulated_Cuboids"
python DNNRegressor-Example.py --training --custom --augment --filter --save /home/hornberger/MasterarbeitTobias/data/simulated/downsampled/h5/Cuboids-Separator-FS7-Augm-Filtered.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/Separator-Final/filteredParticles-Sim-Cuboids-final.json --overrideModel $modelFolder 2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Separator-Final/SimulatedCuboids_Final.txt
#(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV2"


modelFolder="/home/hornberger/MasterarbeitTobias/models/final/simulated/separator-Final/simulated_Zylinder"
python DNNRegressor-Example.py --training --custom --augment --filter --save /home/hornberger/MasterarbeitTobias/data/simulated/downsampled/h5/Zylinder-Separator-FS7-Augm-Filtered.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/Separator-Final/filteredParticles-Sim-Zylinder-final.json --overrideModel $modelFolder 2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Separator-Final/SimulatedZylinder_Final.txt

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV3"


#modelFolder="/home/hornberger/MasterarbeitTobias/models/simulated/separator/FilteredTracks-60kDecaySteps-02Layers-FS7"
#python DNNRegressor-Example.py --training --custom --augment --filter --load /home/hornberger/MasterarbeitTobias/data/simulated/downsampled/h5/Spheres-Separator-FS7-Augm-Filtered.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/SepSimFinetuning/filteredParticles-60kDecaySteps-02Layers-FS7.json --overrideModel $modelFolder  2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Finetuning-Simulated/FilteredParticles-60kDecaySteps-Layers02-FS7.txt

#(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV4"

#modelFolder="/home/hornberger/MasterarbeitTobias/models/simulated/separator/FilteredTracks-60kDecaySteps-02Layers-ReLU"
#python DNNRegressor-Example.py --training --custom --augment --filter --load /home/hornberger/MasterarbeitTobias/data/simulated/downsampled/h5/Spheres-Separator-FS5-Augm-Filtered.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/SepSimFinetuning/filteredParticles-60kDecaySteps-02Layers-ReLU.json --overrideModel $modelFolder  2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Finetuning-Simulated/FilteredParticles-60kDecaySteps-Layers02-ReLU.txt
#(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV5"
