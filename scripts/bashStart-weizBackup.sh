#!/usr/bin/env bash

echo "Starting Execution"
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

python DNNRegressor-Example.py --training --hyperparams ~/MasterarbeitTobias/models/Aug13ev/weiz2/hyperV1.json  --load /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten/Weizen/weizenAug20-train.h5 --overrideModel /home/hornberger/MasterarbeitTobias/models/Aug13ev/weiz5 2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Aug13ev-weiz5.txt

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

#python DNNRegressor-Example.py --training --hyperparams hyperParamFolder/hyperV1.json --load /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten/Weizen/weizenAug13.h5 --overrideModel /home/hornberger/MasterarbeitTobias/models/Aug13ev/weiz2 2>&1 | tee /home/hornberger/MasterarbeitTobias/logs/Aug13ev-weiz1.txt

python DNNRegressor-Example.py --training --hyperparams ~/MasterarbeitTobias/models/Aug13ev/weiz2/hyperV1.json  --load /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten/Weizen/weizenAug20-train.h5 --overrideModel /home/hornberger/MasterarbeitTobias/models/Aug13ev/weiz6 2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Aug13ev-weiz6.txt

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

#python DNNRegressor-Example.py --training --hyperparams hyperParamFolder/hyperV1.json --load /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten/Weizen/weizenAug13.h5 --overrideModel /home/hornberger/MasterarbeitTobias/models/Aug13ev/weiz3 2>&1 | tee /home/hornberger/MasterarbeitTobias/logs/Aug13ev-weiz3.txt

python DNNRegressor-Example.py --training --hyperparams ~/MasterarbeitTobias/models/Aug13ev/weiz2/hyperV1.json  --load /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten/Weizen/weizenAug20-train.h5 --overrideModel /home/hornberger/MasterarbeitTobias/models/Aug13ev/weiz7 2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Aug13ev-weiz7.txt
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

python DNNRegressor-Example.py --training --hyperparams ~/MasterarbeitTobias/models/Aug13ev/weiz2/hyperV1.json  --load /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten/Weizen/weizenAug20-train.h5 --overrideModel /home/hornberger/MasterarbeitTobias/models/Aug13ev/weiz8 2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Aug13ev-weiz8.txt
#python DNNRegressor-Example.py --training --hyperparams hyperParamFolder/hyperV1.json --load /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten/Weizen/weizenAug13.h5 --overrideModel /home/hornberger/MasterarbeitTobias/models/Aug13ev/weiz4 2>&1 | tee /home/hornberger/MasterarbeitTobias/logs/Aug13ev-weiz4.txt
