#!/bin/bash

./omnibench.sh measure saxpy NOPT basic 1000000 -b 8,1024,8
make clean KERNEL=saxpy
./omnibench.sh measure saxpy NOPT basic 10000000 -b 8,1024,8
make clean KERNEL=saxpy
./omnibench.sh measure saxpy NOPT basic 100000000 -b 8,1024,8
make clean KERNEL=saxpy

./omnibench.sh measure saxpy STRIDE basic 1000000 -b 128 -g1,10000,10
make clean KERNEL=saxpy
./omnibench.sh measure saxpy STRIDE basic 10000000 -b 128 -g1,10000,10
make clean KERNEL=saxpy
./omnibench.sh measure saxpy STRIDE basic 100000000 -b 128 -g1,10000,10
make clean KERNEL=saxpy


./omnibench.sh measure matrixMultiply NOPT basic 100 -b 1,32
make clean
./omnibench.sh measure matrixMultiply NOPT basic 1000 -b 1,32
make clean
./omnibench.sh measure matrixMultiply NOPT basic 5000 -b 1,32
make clean
