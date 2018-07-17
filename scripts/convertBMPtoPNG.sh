#!/usr/bin/env bash

targetPrefix='/media/hornberger/data/DatenTobiasH-Juli2018-pngVersion/'

for D in *; do
    if [ -d "${D}" ]; then
        #check if folder exists:
        if [ ! -d "${targetPrefix}$D" ]; then
            echo "creating folder ${targetPrefix}$D"
            mkdir "${targetPrefix}$D"
        fi
        for D2 in ${D}/*; do
            if [ -d "${D2}" ]; then
                #ls "${D2}"
                if [ ! -d "${targetPrefix}$D2" ]; then
                    echo "creating folder ${targetPrefix}$D2"
                    mkdir "${targetPrefix}$D2"
                fi

                echo "mogrify -format png -path ${targetPrefix}$D2 \"$D2/*.bmp\""
                mogrify -format png -path ${targetPrefix}$D2 "$D2/*.bmp"
            fi
        done
    fi
done
