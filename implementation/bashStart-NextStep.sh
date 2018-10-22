#!/usr/bin/env bash

echo "Starting Execution"
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

modelFolder="/home/hornberger/MasterarbeitTobias/models/FineTuning-NextStep/Layer01"
python DNNRegressor-Example.py --training --custom --augment --load /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten2/Kugeln-Band-Juli/kugeln-NextStep-FS5-Baseline.h5  --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/Finetuning-NextStep/Config-Layer01.json --overrideModel $modelFolder  2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Finetuning-NextStep/layer01.txt
(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)


dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV1"

modelFolder="/home/hornberger/MasterarbeitTobias/models/FineTuning-NextStep/Layer05"
python DNNRegressor-Example.py --training --custom --augment --load /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten2/Kugeln-Band-Juli/kugeln-NextStep-FS5-Baseline.h5  --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/Finetuning-NextStep/Config-Layer05.json --overrideModel $modelFolder  2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Finetuning-NextStep/layer05.txt
(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV2"

modelFolder="/home/hornberger/MasterarbeitTobias/models/FineTuning-NextStep/Layer10"
python DNNRegressor-Example.py --training --custom --augment --load /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten2/Kugeln-Band-Juli/kugeln-NextStep-FS5-Baseline.h5  --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/Finetuning-NextStep/Config-Layer10.json --overrideModel $modelFolder  2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Finetuning-NextStep/layer10.txt
(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV3"


#modelFolder="/home/hornberger/MasterarbeitTobias/models/FineTuning/layer/layer15"
#python DNNRegressor-Example.py --training --custom --augment --load /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten2/Kugeln-Band-Juli/kugelnFineTuningLayer.h5  --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/Finetuning/Config-5Layer.json --overrideModel $modelFolder  2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Finetuning/layer15.txt
#(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV4"

