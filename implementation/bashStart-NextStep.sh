#!/usr/bin/env bash

echo "Starting Execution"
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

modelFolder="/home/hornberger/MasterarbeitTobias/models/real/NextStep/NextStep-45kDecaySteps-SpheresReal-Rutsche"
python DNNRegressor-Example.py --training --custom --augment --save /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten2/Kugeln-Rutsche-Sept/h5/kugelnFinal-Rutsche.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/NextStep-Final/RealSpheres-Rutsche-NextStep-45kDecay.json --overrideModel $modelFolder 2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/NextStep-Final/RealSpheres-Rutsche.txt
(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)


dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV1"

modelFolder="/home/hornberger/MasterarbeitTobias/models/real/NextStep/NextStep-45kDecaySteps-PfefferReal-Rutsche"
python DNNRegressor-Example.py --training --custom --augment --save /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten2/Pfeffer-Rutsche-Sept/h5/pfefferFinal-Rutsche.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/NextStep-Final/RealPfeffer-Rutsche-NextStep-45kDecay.json --overrideModel $modelFolder 2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/NextStep-Final/RealPfeffer-Rutsche.txt
(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV2"


modelFolder="/home/hornberger/MasterarbeitTobias/models/real/NextStep/NextStep-45kDecaySteps-ZylinderReal-NoAugm"
python DNNRegressor-Example.py --training --custom --augment --save /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten2/Zylinder-Band-Juli/h5/zylinderFinal-NoAugm.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/NextStep-Final/RealZylinder-NextStep-45kDecay-NoAugm.json  2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/NextStep-Final/RealZylinder-NoAugm.txt
(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV3"


#modelFolder="/home/hornberger/MasterarbeitTobias/models/simulated/NextStep/NextStep-45kDecaySteps-RegL1"
#python DNNRegressor-Example.py --training --custom --augment --load /home/hornberger/MasterarbeitTobias/data/simulated/downsampled/h5/Spheres-FS5-Augm.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/Finetuning-Simulated/Simulated-NextStep-45kDecay-RegL1.json --overrideModel $modelFolder  2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Finetuning-Simulated/45kDecaySteps-RegL1.txt
#(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV4"

