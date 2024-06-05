#!/bin/bash

GPU="RPW6800"
KERNEL="saxpy"
OPT="NOPT"
BENCH_TYPE="blockSizeVar"
make clean
./omnibench.sh measure $KERNEL $OPT 1000000 -b 8,1024,8 --rocprof-only -o results/"$GPU"_"$KERNEL"_"$OPT"_1e6_$BENCH_TYPE.csv
make clean
./omnibench.sh measure $KERNEL $OPT 10000000 -b 8,1024,8 --rocprof-only  -o results/"$GPU"_"$KERNEL"_"$OPT"_1e7_$BENCH_TYPE.csv
make clean
./omnibench.sh measure $KERNEL $OPT 100000000 -b 8,1024,8 --rocprof-only  -o results/"$GPU"_"$KERNEL"_"$OPT"_1e8_$BENCH_TYPE.csv
make clean

BENCH_TYPE="gridSizeVar"
OPT="STRIDE"
./omnibench.sh measure $KERNEL $OPT 1000000 -b 128 -g1,10000,10 --rocprof-only -o results/"$GPU"_"$KERNEL"_"$OPT"_1e6_$BENCH_TYPE.csv
make clean
./omnibench.sh measure $KERNEL $OPT 10000000 -b 128 -g1,10000,10 --rocprof-only -o results/"$GPU"_"$KERNEL"_"$OPT"_1e7_$BENCH_TYPE.csv
make clean
./omnibench.sh measure $KERNEL $OPT 100000000 -b 128 -g1,10000,10 --rocprof-only -o results/"$GPU"_"$KERNEL"_"$OPT"_1e8_$BENCH_TYPE.csv
make clean

KERNEL="matrixMultiply"
BENCH_TYPE="blockSizeVar"
OPT="NOPT"
./omnibench.sh measure $KERNEL $OPT 100 -b 1,32 --rocprof-only -o results/"$GPU"_"$KERNEL"_"$OPT"_100_$BENCH_TYPE.csv
make clean
./omnibench.sh measure $KERNEL $OPT 1000 -b 1,32 --rocprof-only -o results/"$GPU"_"$KERNEL"_"$OPT"_1000_$BENCH_TYPE.csv
make clean
./omnibench.sh measure $KERNEL $OPT 8192 -b 1,32 --rocprof-only -o results/"$GPU"_"$KERNEL"_"$OPT"_8192_$BENCH_TYPE.csv
make clean

BENCH_TYPE="TileSizeVar"
OPT="TILE"
./omnibench.sh measure $KERNEL $OPT 100 -b 1,32 --rocprof-only -o results/"$GPU"_"$KERNEL"_"$OPT"_100_$BENCH_TYPE.csv
make clean
./omnibench.sh measure $KERNEL $OPT 1000 -b 1,32 --rocprof-only -o results/"$GPU"_"$KERNEL"_"$OPT"_100_$BENCH_TYPE.csv
make clean
./omnibench.sh measure $KERNEL $OPT 8192 -b 1,32 --rocprof-only -o results/"$GPU"_"$KERNEL"_"$OPT"_8192_$BENCH_TYPE.csv
make clean
