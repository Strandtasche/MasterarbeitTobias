#!/usr/bin/env bash

echo "Starting Execution"
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

python DNNRegressor-Example.py --training --custom --augment --save /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten2/Kugeln-Band-Juli/kugelnJuli-sep-augmn-v3.h5  --hyperparams hyperParamFolder/hyperKugeln-10-01-separator.json --overrideModel ~/MasterarbeitTobias/models/Sept28/kugeln-Resegmented-sep-aug-v4  2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/kugeln-18-10-09-sep-aug-v4.txt

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV1"

python DNNRegressor-Example.py --training --custom --augment --load /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten2/Kugeln-Band-Juli/kugelnJuli-sep-augmn-v3.h5  --hyperparams hyperParamFolder/hyperKugeln-10-01-separator.json --overrideModel ~/MasterarbeitTobias/models/Sept28/kugeln-Resegmented-sep-aug-v4  2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/kugeln-18-10-09-sep-aug-v4.txt

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV2"

python DNNRegressor-Example.py --training --custom --augment --load /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten2/Kugeln-Band-Juli/kugelnJuli-sep-augmn-v3.h5  --hyperparams hyperParamFolder/hyperKugeln-10-01-separator-linear.json --overrideModel ~/MasterarbeitTobias/models/Sept28/kugeln-Resegmented-sep-aug-linearRegression-v2  2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/kugeln-18-10-09-sep-aug-linear.txt

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV3"

python DNNRegressor-Example.py --training --custom --augment --load /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten2/Kugeln-Band-Juli/kugelnJuli-sep-augmn-v3.h5  --hyperparams hyperParamFolder/hyperKugeln-10-01-separator-linear.json --overrideModel ~/MasterarbeitTobias/models/Sept28/kugeln-Resegmented-sep-aug-linearRegression-v2  2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/kugeln-18-10-09-sep-aug-linear.txt

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV4"

