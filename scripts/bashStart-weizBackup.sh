#!/usr/bin/env bash

echo "Starting Execution"
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

python DNNRegressor-Example.py --training --hyperparams hyperParamFolder/hyperV1.json --save /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten/Weizen/weizenAug13.h5 --overrideModel /home/hornberger/MasterarbeitTobias/models/Aug13ev/weiz1 2>&1 | tee /home/hornberger/MasterarbeitTobias/logs/Aug13ev-weiz1.txt

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

python DNNRegressor-Example.py --training --hyperparams hyperParamFolder/hyperV1.json --load /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten/Weizen/weizenAug13.h5 --overrideModel /home/hornberger/MasterarbeitTobias/models/Aug13ev/weiz2 2>&1 | tee /home/hornberger/MasterarbeitTobias/logs/Aug13ev-weiz1.txt

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

python DNNRegressor-Example.py --training --hyperparams hyperParamFolder/hyperV1.json --load /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten/Weizen/weizenAug13.h5 --overrideModel /home/hornberger/MasterarbeitTobias/models/Aug13ev/weiz3 2>&1 | tee /home/hornberger/MasterarbeitTobias/logs/Aug13ev-weiz3.txt

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

python DNNRegressor-Example.py --training --hyperparams hyperParamFolder/hyperV1.json --load /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten/Weizen/weizenAug13.h5 --overrideModel /home/hornberger/MasterarbeitTobias/models/Aug13ev/weiz4 2>&1 | tee /home/hornberger/MasterarbeitTobias/logs/Aug13ev-weiz4.txt
