#!/usr/bin/env bash

echo "Starting Execution"
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

modelFolder="/home/hornberger/MasterarbeitTobias/models/simulated/separator/FilteredTracks-60kDecaySteps-02Layers-BatchSize100-EarlyStop"
python DNNRegressor-Example.py --training --target 0.00034 --custom --augment --filter --load /home/hornberger/MasterarbeitTobias/data/simulated/downsampled/h5/Spheres-Separator-FS7-Augm-Filtered.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/SepSimFinetuning/filteredParticles-60kDecaySteps-04Layers-FS7-BatchSize100.json --overrideModel $modelFolder 2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Finetuning-Simulated/FilteredParticles-60kDecaySteps-04Layers-FS7-BatchSize100-EarlyStop.txt
#(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV1"

modelFolder="/home/hornberger/MasterarbeitTobias/models/simulated/separator/FilteredTracks-60kDecaySteps-02Layers-BatchSize100-L1"
python DNNRegressor-Example.py --training --custom --augment --filter --load /home/hornberger/MasterarbeitTobias/data/simulated/downsampled/h5/Spheres-Separator-FS7-Augm-Filtered.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/SepSimFinetuning/filteredParticles-60kDecaySteps-04Layers-FS7-BatchSize100-L1.json --overrideModel $modelFolder 2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Finetuning-Simulated/FilteredParticles-60kDecaySteps-04Layers-FS7-BatchSize100-L1.txt
#(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV2"

modelFolder="/home/hornberger/MasterarbeitTobias/models/simulated/separator/FilteredTracks-60kDecaySteps-02Layers-BatchSize100-L2"
python DNNRegressor-Example.py --training --custom --augment --filter --load /home/hornberger/MasterarbeitTobias/data/simulated/downsampled/h5/Spheres-Separator-FS7-Augm-Filtered.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/SepSimFinetuning/filteredParticles-60kDecaySteps-04Layers-FS7-BatchSize100-L2.json --overrideModel $modelFolder 2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Finetuning-Simulated/FilteredParticles-60kDecaySteps-04Layers-FS7-BatchSize100-L2.txt
#(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)

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
