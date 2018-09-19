#!/usr/bin/env bash

targetPrefix='/media/hornberger/data/test2/'

set -e

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

                mogrify -format png -size 2320x1728 -depth 8 -path ${targetPrefix}$D2 "gray:$D2/*.raw"
            fi
        done
    fi

done
