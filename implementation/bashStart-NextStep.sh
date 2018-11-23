#!/usr/bin/env bash

echo "Starting Execution"
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

modelFolder="/home/hornberger/MasterarbeitTobias/models/simulated/NextStep/NextStep-45kDecaySteps-Batchsize250"
python DNNRegressor-Example.py --training --custom --augment --load /home/hornberger/MasterarbeitTobias/data/simulated/downsampled/h5/Spheres-FS5-Augm.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/Finetuning-Simulated/Simulated-NextStep-45kDecay-Batchsize250.json  --overrideModel $modelFolder  2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Finetuning-Simulated/45kDecaySteps-Batchsize250.txt
#(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)


dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV1"

#modelFolder="/home/hornberger/MasterarbeitTobias/models/simulated/NextStep/NextStep-45kDecaySteps-RegL2"
#python DNNRegressor-Example.py --training --custom --augment --load /home/hornberger/MasterarbeitTobias/data/simulated/downsampled/h5/Spheres-FS5-Augm.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/Finetuning-Simulated/Simulated-NextStep-45kDecay-RegL2.json --overrideModel $modelFolder  2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Finetuning-Simulated/45kDecaySteps-RegL2.txt
#(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV2"

#modelFolder="/home/hornberger/MasterarbeitTobias/models/simulated/NextStep/NextStep-25kDecaySteps"
#python DNNRegressor-Example.py --training --custom --augment --load /home/hornberger/MasterarbeitTobias/data/simulated/downsampled/h5/Spheres-FS5-Augm.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/Finetuning-Simulated/Simulated-NextStep-25kDecay.json  --overrideModel $modelFolder  2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Finetuning-Simulated/25kDecaySteps.txt
#(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV3"


#modelFolder="/home/hornberger/MasterarbeitTobias/models/simulated/NextStep/NextStep-45kDecaySteps-RegL1"
#python DNNRegressor-Example.py --training --custom --augment --load /home/hornberger/MasterarbeitTobias/data/simulated/downsampled/h5/Spheres-FS5-Augm.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/Finetuning-Simulated/Simulated-NextStep-45kDecay-RegL1.json --overrideModel $modelFolder  2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Finetuning-Simulated/45kDecaySteps-RegL1.txt
#(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV4"

