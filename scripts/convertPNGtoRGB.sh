#!/usr/bin/env bash

targetPrefix='/media/hornberger/data/Daten_Tobias-September2018-pngVersion-debayered/'

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
                if [ ! -d "${targetPrefix}$D2" ]; then
                    echo "creating folder ${targetPrefix}$D2"
                    mkdir "${targetPrefix}$D2"
                fi

                echo "python ~/MasterarbeitTobias/scripts -i "$D2" -o "${targetPrefix}$D2/""
                python3 ~/MasterarbeitTobias/scripts/debayer.py -i "$D2" -o "${targetPrefix}$D2/"
                #pip freeze | grep opencv
            fi
        done
    fi
done
