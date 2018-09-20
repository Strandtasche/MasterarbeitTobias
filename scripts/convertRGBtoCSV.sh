#!/usr/bin/env bash

targetPrefix='/media/hornberger/data/Daten_Hornberger_September2018-pngVersion-debayered-csved/'
#targetPrefix='/media/hornberger/data/Daten-Kronauer-September2018-pngVersion-debayered-csved/'

for D in *; do
    if [ -d "${D}" ]; then
        #check if folder exists:
        if [ ! -d "${targetPrefix}$D" ]; then
            echo "creating folder ${targetPrefix}$D"
            mkdir -p "${targetPrefix}$D"
        fi
        for D2 in ${D}/*; do
            if [ -d "${D2}" ]; then
                #ls "${D2}"
                files=( ${D2}/*.png )
                origin="${files[0]//00000/\%05d}"
                #| sed -e 's/00000/%05d/g'
                echo "python ~/MasterarbeitTobias/scripts/segment.py -i "$origin" -o "${targetPrefix}$D2"" -nd
                python ~/MasterarbeitTobias/scripts/segment.py -i "$origin" -o "${targetPrefix}$D2" -nd
                #python ~/MasterarbeitTobias/scripts/debayer.py -i "$D2" -o "${targetPrefix}$D2/"
                #pip freeze | grep opencv
            fi
        done
    fi
done
