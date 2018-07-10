find /media/hornberger/data/Tracksort/Daten/ -type f -name "*data.csv" -print0 | while IFS= read -r -d $'\0' line; do
    echo "$line"
    laa=${line:39}
    baa=${laa/ /_}
    foo=${baa////_}
    final="./data_extracted/$foo"
    echo "$foo"
    mv "${line/ /\ }" "$final"
done
