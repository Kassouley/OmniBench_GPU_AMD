#!/bin/bash

./omnibench.sh measure saxpy NOPT basic 1000000 -b 8,1024,8
./omnibench.sh measure saxpy NOPT basic 10000000 -b 8,1024,8
./omnibench.sh measure saxpy NOPT basic 100000000 -b 8,1024,8

./omnibench.sh measure saxpy STRIDE basic 1000000 -b 128 -g1,10000,10
./omnibench.sh measure saxpy STRIDE basic 10000000 -b 128 -g1,10000,10
./omnibench.sh measure saxpy STRIDE basic 100000000 -b 128 -g1,10000,10