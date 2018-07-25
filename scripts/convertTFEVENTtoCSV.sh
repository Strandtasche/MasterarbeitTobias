# Make sure globstar is enabled
shopt -s globstar
for i in **/*tfevents*; do # Whitespace-safe and recursive
    DIR=$(dirname "$i")
    if [[ $DIR != *eval ]]; then
        python ~/MasterarbeitTobias/scripts/exportTensorFlowLog.py $i $DIR
        #mv "$DIR/scalar.csv" "$DIR/$(basename $i).csv"
    fi
done


