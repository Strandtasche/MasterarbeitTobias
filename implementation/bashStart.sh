#!/usr/bin/env bash

echo "Starting Execution"
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

python DNNRegressor-Example.py --fake --plot --hyperparams hyperParamsBatch1000.json

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished batch 1000"

python DNNRegressor-Example.py --fake --plot --hyperparams hyperParamsBatch2000.json

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished batch 2000"

python DNNRegressor-Example.py --fake --plot --hyperparams hyperParamsBatch3000.json

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished batch 3000"

python DNNRegressor-Example.py --fake --plot --hyperparams hyperParamsBatch5000.json

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished batch 5000"

