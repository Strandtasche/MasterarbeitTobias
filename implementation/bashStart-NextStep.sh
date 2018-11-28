#!/usr/bin/env bash

echo "Starting Execution"
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

modelFolder="/home/hornberger/MasterarbeitTobias/models/real/NextStep/NextStep-45kDecaySteps-WeizenReal"
python DNNRegressor-Example.py --training --custom --augment --save /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten2/Weizen-Band-Juli/h5/weizenFinal.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/NextStep-Final/RealWeizen-NextStep-45kDecay.json --overrideModel $modelFolder 2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/NextStep-Final/RealWeizen.txt
(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)


dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV1"

modelFolder="/home/hornberger/MasterarbeitTobias/models/real/NextStep/NextStep-45kDecaySteps-SpheresReal"
python DNNRegressor-Example.py --training --custom --augment --save /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten2/Kugeln-Band-Juli/h5/kugelnFinal.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/NextStep-Final/RealSpheres-NextStep-45kDecay.json --overrideModel $modelFolder 2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/NextStep-Final/RealSpheres.txt
(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV2"

modelFolder="/home/hornberger/MasterarbeitTobias/models/real/NextStep/NextStep-45kDecaySteps-PfefferReal"
python DNNRegressor-Example.py --training --custom --augment --save /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten2/Pfeffer-Band-Juli/h5/pfefferFinal.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/NextStep-Final/RealPfeffer-NextStep-45kDecay.json --overrideModel $modelFolder 2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/NextStep-Final/RealPfeffer.txt
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

