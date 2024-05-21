#!/bin/bash
TMP_PATH='./tmp'
ASM_PATH='./asm'
OUT_PATH='./output'
KERNELS=$1

create_output_file()
{
        echo -n "NTHREAD" > $1
        for i in $values ; do
                echo -n ", $i, $i, $i" >> $1
        done
        echo "" >> $1
}

for KERNEL in $KERNELS; do
        cd $KERNEL
        make $KERNEL
        mkdir $TMP_PATH
        mkdir $OUT_PATH
        mkdir $ASM_PATH

        values=$(head -n 1 $KERNEL.config | tail -n 1)
        block_sizes=$(head -n 2 $KERNEL.config | tail -n 1)
        
       create_output_file "$OUT_PATH/first_output.csv"
       create_output_file "$OUT_PATH/last_output.csv"

        for block_size in $block_sizes ; do
                echo -n "$block_size" >> "$OUT_PATH/first_output.csv"
                echo -n "$block_size" >> "$OUT_PATH/last_output.csv"
                for val in $values; do
                        echo "========================================================="
                        echo "Executing $KERNEL with size=$val and block_size=$block_size"
                        rocprof -o $TMP_PATH/results.csv -i input.txt --stats --basenames on  $KERNEL "$val" "$block_size" 50 > /dev/null
                        first_dur=$(cat $TMP_PATH/results.csv | awk -F ',' 'NR == 2 {print $30}')
                        first_wavefront=$(cat $TMP_PATH/results.csv | awk -F ',' 'NR == 2 {print $21}')
                        first_L2_hit=$(cat $TMP_PATH/results.csv | awk -F ',' 'NR == 2 {print $22}')
                        last_dur=$(cat $TMP_PATH/results.csv | awk -F ',' 'NR == 50 {print $30}')
                        last_wavefront=$(cat $TMP_PATH/results.csv | awk -F ',' 'NR == 50 {print $21}')
                        last_L2_hit=$(cat $TMP_PATH/results.csv | awk -F ',' 'NR == 50 {print $22}')
                        echo "rocprof duration   : $first_dur ns | $last_dur ns"
                        echo "rocprof wavefront  : $first_wavefront | $last_wavefront"
                        echo "rocprof cacheL2hit : $first_L2_hit | $last_L2_hit"
                        echo -n ", $first_dur, $first_wavefront, $first_L2_hit" >> "$OUT_PATH/first_output.csv"
                        echo -n ", $last_dur, $last_wavefront, $last_L2_hit" >> "$OUT_PATH/last_output.csv"
                        rm $TMP_PATH/results*
                done
                echo "" >> "$OUT_PATH/first_output.csv"
                echo "" >> "$OUT_PATH/last_output.csv"
        done
        rm tmp -r

        roc-obj -o $ASM_PATH -d $KERNEL
done
