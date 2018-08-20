#!/usr/bin/env bash

echo "Starting Execution"
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

python DNNRegressor-Example.py --training --hyperparams ~/MasterarbeitTobias/models/Aug11ev/kug1/hyperV1.json --save ~/MasterarbeitTobias/data/selbstgesammelteDaten/Kugeln/kugelnAug11.H5 --overrideModel ~/MasterarbeitTobias/models/Aug11ev/kug5 2>&1 | tee -a ~/MasterarbeitTobias/logs/Aug11ev-KUG5.txt

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

python DNNRegressor-Example.py --training --hyperparams ~/MasterarbeitTobias/models/Aug11ev/kug1/hyperV1.json --load ~/MasterarbeitTobias/data/selbstgesammelteDaten/Kugeln/kugelnAug11.H5 --overrideModel ~/MasterarbeitTobias/models/Aug11ev/kug6 2>&1 | tee -a ~/MasterarbeitTobias/logs/Aug11ev-KUG6.txt

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

python DNNRegressor-Example.py --training --hyperparams ~/MasterarbeitTobias/models/Aug11ev/kug1/hyperV1.json --load ~/MasterarbeitTobias/data/selbstgesammelteDaten/Kugeln/kugelnAug11.H5 --overrideModel ~/MasterarbeitTobias/models/Aug11ev/kug7 2>&1 | tee ~/MasterarbeitTobias/logs/Aug11ev-KUG7.txt

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

python DNNRegressor-Example.py --training --hyperparams ~/MasterarbeitTobias/models/Aug11ev/kug1/hyperV1.json --load ~/MasterarbeitTobias/data/selbstgesammelteDaten/Kugeln/kugelnAug11.H5 --overrideModel ~/MasterarbeitTobias/models/Aug11ev/kug8 2>&1 | tee ~/MasterarbeitTobias/logs/Aug11ev-KUG8.txt
