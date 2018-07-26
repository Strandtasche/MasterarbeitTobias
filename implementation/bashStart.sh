#!/usr/bin/env bash

echo "Starting Execution"
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

python DNNRegressor-Example.py --training

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV1"

python DNNRegressor-Example.py --training --custom

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV2"

python DNNRegressor-Example.py --training --overwriteModel /home/hornberger/MasterarbeitTobias/models/customComp3/Alter-Premade

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV3"

python DNNRegressor-Example.py --training --custom --overwriteModel /home/hornberger/MasterarbeitTobias/models/customComp3/Alter-custom

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV4"

