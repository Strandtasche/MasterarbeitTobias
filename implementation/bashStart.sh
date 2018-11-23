#!/usr/bin/env bash

echo "Starting Execution"
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

modelFolder="/home/hornberger/MasterarbeitTobias/models/simulated/separator/FilteredTracks-60kDecaySteps-02Layers"
python DNNRegressor-Example.py --training --custom --augment --filter --load /home/hornberger/MasterarbeitTobias/data/simulated/downsampled/h5/Spheres-Separator-FS5-Augm-Filtered.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/SepSimFinetuning/filteredParticles-60kDecaySteps-02Layers.json --overrideModel $modelFolder  2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Finetuning-Simulated/FilteredParticles-60kDecaySteps-02Layers.txt
#(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)


dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV1"

modelFolder="/home/hornberger/MasterarbeitTobias/models/simulated/separator/FilteredTracks-60kDecaySteps-05Layers"
python DNNRegressor-Example.py --training --custom --augment --filter --load /home/hornberger/MasterarbeitTobias/data/simulated/downsampled/h5/Spheres-Separator-FS5-Augm-Filtered.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/SepSimFinetuning/filteredParticles-RegL2.json --overrideModel $modelFolder  2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Finetuning-Simulated/FilteredParticles-RegL2.txt
#(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV2"

modelFolder="/home/hornberger/MasterarbeitTobias/models/simulated/separator/FilteredTracks-Regression3Layers-60kDecaySteps"
python DNNRegressor-Example.py --training --custom --augment --filter --load /home/hornberger/MasterarbeitTobias/data/simulated/downsampled/h5/Spheres-Separator-FS5-Augm-Filtered.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/SepSimFinetuning/filteredParticles-Regression3Layers-60kDecaySteps.json --overrideModel $modelFolder  2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Finetuning-Simulated/FilteredParticles-Regression3Layer-60kDecaySteps.txt
#(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV3"


modelFolder="/home/hornberger/MasterarbeitTobias/models/simulated/separator/FilteredTracks-60kDecaySteps-FS3"
python DNNRegressor-Example.py --training --custom --augment --filter --save /home/hornberger/MasterarbeitTobias/data/simulated/downsampled/h5/Spheres-Separator-FS3-Augm-Filtered.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/SepSimFinetuning/filteredParticles-60kDecaySteps-FS3.json --overrideModel $modelFolder  2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Finetuning-Simulated/FilteredParticles-FeatureSize3.txt
#(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV4"

modelFolder="/home/hornberger/MasterarbeitTobias/models/simulated/separator/FilteredTracks-60kDecaySteps-FS7"
python DNNRegressor-Example.py --training --custom --augment --filter --save /home/hornberger/MasterarbeitTobias/data/simulated/downsampled/h5/Spheres-Separator-FS7-Augm-Filtered.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/SepSimFinetuning/filteredParticles-60kDecaySteps-FS7.json --overrideModel $modelFolder  2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Finetuning-Simulated/FilteredParticles-FeatureSize7.txt
#(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV5"
