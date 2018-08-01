
echo "file to be split: $1"

numberColumns="$(head -1 "$1" | sed 's/[^,]//g' | wc -c)"
last=0
if [ ! -d "$2" ]; then
    echo "creating folder $2"
    mkdir "$2"
fi
for i in `seq -f "%05g" 1 400 $numberColumns`; do
    cut -d "," -f$((10#$i))-$((10#$i + 399)) $1 > "$2/${1: :-4}_$i.csv"
done
