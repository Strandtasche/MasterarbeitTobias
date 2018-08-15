#!/usr/bin/env bash

echo "Starting Execution"
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

python DNNRegressor-Example.py --training --hyperparams hyperParamFolder/hyperV1.json --save /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten/Pfeffer/pfefferAug14.h5 --overrideModel /home/hornberger/MasterarbeitTobias/models/Aug14ev/pfef1 2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Aug14ev-pfef1.txt

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

python DNNRegressor-Example.py --training --hyperparams hyperParamFolder/hyperV1.json --load /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten/Pfeffer/pfefferAug14.h5 --overrideModel /home/hornberger/MasterarbeitTobias/models/Aug14ev/pfef2 2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Aug14ev-pfef2.txt

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

python DNNRegressor-Example.py --training --hyperparams hyperParamFolder/hyperV1.json --load /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten/Pfeffer/pfefferAug14.h5 --overrideModel /home/hornberger/MasterarbeitTobias/models/Aug14ev/pfef3 2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Aug14ev-pfef3.txt

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

python DNNRegressor-Example.py --training --hyperparams hyperParamFolder/hyperV1.json --load /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten/Pfeffer/pfefferAug14.h5 --overrideModel /home/hornberger/MasterarbeitTobias/models/Aug14ev/pfef4 2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Aug14ev-pfef4.txt
