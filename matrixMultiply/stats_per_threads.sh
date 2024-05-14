#!/bin/bash

output_file="measure_tmp.out"

rm $output_file
echo "SIZE, NB_THREAD, MED, MIN, AVG, SCALABILITY" > $output_file

make saxpy

values=("500000" "1000000" "1048576" "1500000" "2000000" "3000000" "4000000" "5000000" "10000000")
block_sizes=("8" "16" "32" "64" "128" "256" "512" "1024")

for val in "${values[@]}"; do
    for block_size in "${block_sizes[@]}" ; do
        echo "Executing saxpy with size=$val and block_size=$block_size"
        ./saxpy "$val" "$block_size" 10000 1000
    done
done


echo "NTHREAD,$(IFS=,; echo "${values[*]}")" > output.csv

for block_size in "${block_sizes[@]}" ; do
    row="$block_size"
    for val in "${values[@]}"; do
        result=$(awk -F',' '$1=='$val' && $2=='$block_size' {print $5}' $output_file)
        row="$row,$result"
    done

    echo "$row" >> output.csv
done
