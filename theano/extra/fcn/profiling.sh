#!/bin/sh

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR


# run theano GPU profiling

mkdir -p theano_profiling

CUDA_LAUNCH_BLOCKING=1 THEANO_FLAGS=profile=1 python fcn5_keras.py > theano_profiling/fcn5_keras.log 2>&1
CUDA_LAUNCH_BLOCKING=1 THEANO_FLAGS=profile=1 python fcn5_raw_theano.py > theano_profiling/fcn5_raw_theano.log 2>&1
CUDA_LAUNCH_BLOCKING=1 THEANO_FLAGS=profile=1 python ../../benchmark.py -a fcn5 -B 64 -n 190 > theano_profiling/fcn5_lasagne.log 2>&1



mkdir -p nvprof_profiling

nvprof python fcn5_keras.py > nvprof_profiling/fcn5_keras.log 2>&1
nvprof python fcn5_raw_theano.py > nvprof_profiling/fcn5_raw_theano.log 2>&1
nvprof python ../../benchmark.py -a fcn5 -B 64 -n 190 > nvprof_profiling/fcn5_lasagne.log 2>&1

