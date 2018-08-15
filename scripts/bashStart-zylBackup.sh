#!/usr/bin/env bash

echo "Starting Execution"
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

python DNNRegressor-Example.py --training --hyperparams hyperParamFolder/hyperV1.json --save ~/MasterarbeitTobias/data/selbstgesammelteDaten/Zylinder/zylinderAug10.h5 --overrideModel ~/MasterarbeitTobias/models/Aug10Ev/zyl1 2>&1 | tee ~/MasterarbeitTobias/logs/Auf10Ev-zyl1.txt

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

python DNNRegressor-Example.py --training --hyperparams hyperParamFolder/hyperV1.json --load ~/MasterarbeitTobias/data/selbstgesammelteDaten/Zylinder/zylinderAug10.h5 --overrideModel ~/MasterarbeitTobias/models/Aug10Ev/zyl2 2>&1 | tee ~/MasterarbeitTobias/logs/Auf10Ev-zyl2.txt

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

python DNNRegressor-Example.py --training --hyperparams hyperParamFolder/hyperV1.json --load ~/MasterarbeitTobias/data/selbstgesammelteDaten/Zylinder/zylinderAug10.h5 --overrideModel ~/MasterarbeitTobias/models/Aug10Ev/zyl3 2>&1 | tee ~/MasterarbeitTobias/logs/Auf10Ev-zyl3.txt

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

python DNNRegressor-Example.py --training --hyperparams hyperParamFolder/hyperV1.json --load ~/MasterarbeitTobias/data/selbstgesammelteDaten/Zylinder/zylinderAug10.h5 --overrideModel ~/MasterarbeitTobias/models/Aug10Ev/zyl4 2>&1 | tee ~/MasterarbeitTobias/logs/Auf10Ev-zyl4.txt

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

