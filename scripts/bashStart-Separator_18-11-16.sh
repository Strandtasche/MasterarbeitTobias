#!/usr/bin/env bash

echo "Starting Execution"
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

modelFolder="/home/hornberger/MasterarbeitTobias/models/simulated/separator/FilteredTracks-RegL1"
python DNNRegressor-Example.py --training --custom --augment --load /home/hornberger/MasterarbeitTobias/data/simulated/downsampled/h5/Spheres-Separator-FS5-Augm-Filtered.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/SepSimFinetuning/filteredParticles-RegL1.json --overrideModel $modelFolder  2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Finetuning-Simulated/FilteredParticles-RegL1.txt
#(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)


dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV1"

modelFolder="/home/hornberger/MasterarbeitTobias/models/simulated/separator/FilteredTracks-RegL2"
python DNNRegressor-Example.py --training --custom --augment --load /home/hornberger/MasterarbeitTobias/data/simulated/downsampled/h5/Spheres-Separator-FS5-Augm-Filtered.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/SepSimFinetuning/filteredParticles-RegL2.json --overrideModel $modelFolder  2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Finetuning-Simulated/FilteredParticles-RegL2.txt
#(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV2"

modelFolder="/home/hornberger/MasterarbeitTobias/models/simulated/separator/FilteredTracks-60kDecaySteps"
python DNNRegressor-Example.py --training --custom --augment --load /home/hornberger/MasterarbeitTobias/data/simulated/downsampled/h5/Spheres-Separator-FS5-Augm-Filtered.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/SepSimFinetuning/filteredParticles-60kDecaySteps.json --overrideModel $modelFolder  2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Finetuning-Simulated/FilteredParticles-60kDecaySteps.txt
#(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV3"


modelFolder="/home/hornberger/MasterarbeitTobias/models/simulated/separator/FilteredTracks-BatchSize1000"
python DNNRegressor-Example.py --training --custom --augment --load /home/hornberger/MasterarbeitTobias/data/simulated/downsampled/h5/Spheres-Separator-FS5-Augm-Filtered.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/SepSimFinetuning/filteredParticles-BatchSize1000.json --overrideModel $modelFolder  2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Finetuning-Simulated/FilteredParticles-BatchSize1000.txt
#(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV4"

