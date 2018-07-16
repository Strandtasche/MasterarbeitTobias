#!/usr/bin/env bash

if [ ! -d ~/extracted_blobs/Pfeffer ]; then
    echo "Folder not found, creating folder"
    mkdir -p  ~/extracted_blobs/Pfeffer/
else
    echo "folder found"
fi

for d in `ls -d /media/hornberger/data/DatenTobiasH-Juli2018/Pfeffer/*`; do
    outputDir=~/extracted_blobs/Pfeffer/
    python ~/Masterarbeit/scripts/extract_blobs.py --input_directory $d --file_extension bmp --output_directory $outputDir
done
