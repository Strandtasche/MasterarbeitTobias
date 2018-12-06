#!/usr/bin/env bash

echo "Starting Execution"
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

python DNNRegressor-Example.py --training --custom --augment --filter --save /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten2/Kugeln-Band-Juli/h5/kugelnSeparatorFinal-FS7-filterAugm-750Distance.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/Separator-Final/filteredParticles-Real-Spheres-final.json 2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Separator-Final/RealSpheres_Final.txt

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV1"

python DNNRegressor-Example.py --training --custom --augment --filter --load /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten2/Pfeffer-Band-Juli/h5/pfefferSeparatorFinal-FS7-filterAugm-750Distance.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/Separator-Final/filteredParticles-Real-Pfeffer-final.json 2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Separator-Final/RealPfeffer_Final.txt

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV2"


python DNNRegressor-Example.py --training --custom --augment --filter --save /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten2/Zylinder-Band-Juli/h5/zylinderSeparatorFinal-FS7-filterAugm-750Distance.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/Separator-Final/filteredParticles-Real-Zylinder-final.json 2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Separator-Final/RealZylinder_Final.txt

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV3"


python DNNRegressor-Example.py --training --custom --augment --filter --save /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten2/Weizen-Band-Juli/h5/weizenSeparatorFinal-FS7-filterAugm-750Distance.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/Separator-Final/filteredParticles-Real-Weizen-final.json 2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Separator-Final/RealWeizen_Final.txt

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV4"

python DNNRegressor-Example.py --training --custom --augment --filter --load /home/hornberger/MasterarbeitTobias/data/selbstgesammelteDaten2/Kugeln-Band-Juli/h5/kugelnSeparatorFinal-FS7-filterAugm-750Distance.h5 --hyperparams /home/hornberger/Projects/MasterarbeitTobias/implementation/Separator-Final/filteredParticles-Real-Spheres-NoReg.json 2>&1 | tee -a /home/hornberger/MasterarbeitTobias/logs/Separator-Final/RealSpheres_NoReg.txt

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
echo "finished hyperV5"
