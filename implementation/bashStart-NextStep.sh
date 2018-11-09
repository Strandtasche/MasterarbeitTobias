#!/usr/bin/env bash

echo "Starting Execution"
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

modelFolder="/home/hornberger/MasterarbeitTobias/models/simulated/NextStep/NextStep-60kDecaySteps-RegL1"
python DNNRegressor-Example.py --training --custom --augment --load /home/hornberger/MasterarbeitTobias/data/simulated/downsampled/h5/Spheres-FS5-Augm.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/Finetuning-Simulated/Simulated-NextStep-60kDecay-RegL1.json  --overrideModel $modelFolder  2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Finetuning-Simulated/60kDecaySteps-RegL1.txt
(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)


dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV1"

modelFolder="/home/hornberger/MasterarbeitTobias/models/simulated/NextStep/NextStep-60kDecaySteps-RegL2"
python DNNRegressor-Example.py --training --custom --augment --load /home/hornberger/MasterarbeitTobias/data/simulated/downsampled/h5/Spheres-FS5-Augm.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/Finetuning-Simulated/Simulated-NextStep-60kDecay-RegL2.json  --overrideModel $modelFolder  2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Finetuning-Simulated/60kDecaySteps-RegL2.txt
(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV2"

#modelFolder="/home/hornberger/MasterarbeitTobias/models/simulated/NextStep/NextStep-60kDecaySteps-10Layers-RegL2"
#python DNNRegressor-Example.py --training --custom --augment --load /home/hornberger/MasterarbeitTobias/data/simulated/downsampled/h5/Spheres-FS5-Augm.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/Finetuning-Simulated/Simulated-NextStep-60kDecay-10Layers-RegL2.json  --overrideModel $modelFolder  2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Finetuning-Simulated/60kDecaySteps-10layers-RegL2.txt
#(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV3"


#modelFolder="/home/hornberger/MasterarbeitTobias/models/simulated/NextStep/NextStep-ReLU"
#python DNNRegressor-Example.py --training --custom --augment --save /home/hornberger/MasterarbeitTobias/data/simulated/downsampled/h5/Spheres-FS5-Augm.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/Finetuning-Simulated/Simulated-NextStep-60kDecay-ReLU.json --overrideModel $modelFolder  2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Finetuning-Simulated/NextStep-60kDecay-ReLU.json
#(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV4"

