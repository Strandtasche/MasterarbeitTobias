#!/usr/bin/env bash

echo "Starting Execution"
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

#modelFolder="/home/hornberger/MasterarbeitTobias/models/FineTuning/Decay/Baseline"
#python DNNRegressor-Example.py --training --custom --augment --save /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten2/Kugeln-Band-Juli/kugelnFineTuningLayer.h5  --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/Finetuning/ConfigBaseline.json --overrideModel $modelfolder  2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Finetuning/Baseline.txt
#(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)


dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV1"

#python DNNRegressor-Example.py --training --custom --augment --load /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten2/Kugeln-Band-Juli/kugelnFineTuningLayer.h5  --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/Finetuning/Config-1Layer.json --overrideModel ~/MasterarbeitTobias/models/FineTuning/layer/layer01  2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Finetuning/layer01.txt
#(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV2"

modelFolder="/home/hornberger/MasterarbeitTobias/models/FineTuning/layer/layer10"
python DNNRegressor-Example.py --training --custom --augment --load /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten2/Kugeln-Band-Juli/kugelnFineTuningLayer.h5  --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/Finetuning/Config-10Layer.json --overrideModel $modelFolder  2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Finetuning/layer10.txt
(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV3"


modelFolder="/home/hornberger/MasterarbeitTobias/models/FineTuning/layer/layer15"
python DNNRegressor-Example.py --training --custom --augment --load /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten2/Kugeln-Band-Juli/kugelnFineTuningLayer.h5  --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/Finetuning/Config-5Layer.json --overrideModel $modelFolder  2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Finetuning/layer15.txt
(cd $modelFolder && bash /home/hornberger/MasterarbeitTobias/scripts/convertTFEVENTtoCSV.sh delete)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV4"

