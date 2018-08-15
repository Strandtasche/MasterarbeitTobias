#!/usr/bin/env bash

echo "Starting Execution"
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

python DNNRegressor-Example.py --training --hyperparams hyperParamFolder/hyperV1.json --save ~/MasterarbeitTobias/data/selbstgesammelteDaten/Kugeln/kugelnAug11.H5 --overrideModel ~/MasterarbeitTobias/models/Aug11ev/kug1 2>&1 | tee ~/MasterarbeitTobias/logs/Aug11ev-KUG1.txt

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

python DNNRegressor-Example.py --training --hyperparams hyperParamFolder/hyperV1.json --save ~/MasterarbeitTobias/data/selbstgesammelteDaten/Kugeln/kugelnAug11.H5 --overrideModel ~/MasterarbeitTobias/models/Aug11ev/kug2 2>&1 | tee ~/MasterarbeitTobias/logs/Aug11ev-KUG2.txt

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

python DNNRegressor-Example.py --training --hyperparams hyperParamFolder/hyperV1.json --save ~/MasterarbeitTobias/data/selbstgesammelteDaten/Kugeln/kugelnAug11.H5 --overrideModel ~/MasterarbeitTobias/models/Aug11ev/kug3 2>&1 | tee ~/MasterarbeitTobias/logs/Aug11ev-KUG3.txt

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

python DNNRegressor-Example.py --training --hyperparams hyperParamFolder/hyperV1.json --save ~/MasterarbeitTobias/data/selbstgesammelteDaten/Kugeln/kugelnAug11.H5 --overrideModel ~/MasterarbeitTobias/models/Aug11ev/kug4 2>&1 | tee ~/MasterarbeitTobias/logs/Aug11ev-KUG4.txt
