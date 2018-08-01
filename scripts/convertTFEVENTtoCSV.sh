# Make sure globstar is enabled
shopt -s globstar

set -e
for i in **/*tfevents*; do # Whitespace-safe and recursive
    DIR=$(dirname "$i")
    if [[ $DIR != *eval ]]; then
        python ~/MasterarbeitTobias/scripts/exportTensorFlowLog.py $i $DIR
        mv "$DIR/scalars.csv" "$DIR/$(basename $i).csv"
        if [ $1== 'delete']; then
            echo "rm $i"
        fi
    fi
done
