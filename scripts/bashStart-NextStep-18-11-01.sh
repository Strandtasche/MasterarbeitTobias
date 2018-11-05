#!/usr/bin/env bash

echo "Starting Execution"
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

modelFolder="/home/hornberger/MasterarbeitTobias/models/simulated/NextStep/NextStep-60kDecaySteps-01Layers"
python DNNRegressor-Example.py --training --custom --augment --save /home/hornberger/MasterarbeitTobias/data/simulated/downsampled/h5/Spheres-FS5-Augm.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/Finetuning-Simulated/Simulated-NextStep-60kDecay-01Layers.json  --overrideModel $modelFolder  2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Finetuning-Simulated/60kDecaySteps-01layers.txt
(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)


dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV1"

modelFolder="/home/hornberger/MasterarbeitTobias/models/simulated/NextStep/NextStep-60kDecaySteps-05Layers"
python DNNRegressor-Example.py --training --custom --augment --load /home/hornberger/MasterarbeitTobias/data/simulated/downsampled/h5/Spheres-FS5-Augm.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/Finetuning-Simulated/Simulated-NextStep-60kDecay-05Layers.json  --overrideModel $modelFolder  2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Finetuning-Simulated/60kDecaySteps-05layers.txt
(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV2"

modelFolder="/home/hornberger/MasterarbeitTobias/models/simulated/NextStep/NextStep-FS3"
python DNNRegressor-Example.py --training --custom --augment --save /home/hornberger/MasterarbeitTobias/data/simulated/downsampled/h5/Spheres-FS3-Augm.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/Finetuning-Simulated/Simulated-NextStep-FS3.json --overrideModel $modelFolder  2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Finetuning-Simulated/NextStep-FS3.json
(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV3"


modelFolder="/home/hornberger/MasterarbeitTobias/models/simulated/NextStep/NextStep-FS10"
python DNNRegressor-Example.py --training --custom --augment --save /home/hornberger/MasterarbeitTobias/data/simulated/downsampled/h5/Spheres-FS10-Augm.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/Finetuning-Simulated/Simulated-NextStep-FS10.json --overrideModel $modelFolder  2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Finetuning-Simulated/NextStep-FS10.json
(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV4"

