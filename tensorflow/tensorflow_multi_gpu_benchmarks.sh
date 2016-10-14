#!/bin/bash

CMBS=16
CNB=100

RMBS=128
RNB=100

FMBS=8192
FNB=100

for i in "$@"
do
case $i in
    -cm=*|--cnn_mini_batch_size=*)
    CMBS="${i#*=}"
    shift # past argument=value
    ;;
    -cn=*|--cnn_num_batch=*)
    CNB="${i#*=}"
    shift # past argument=value
    ;;

    -rm=*|--rnn_mini_batch_size=*)
    RMBS="${i#*=}"
    shift # past argument=value
    ;;
    -rn=*|--rnn_num_batch=*)
    RNB="${i#*=}"
    shift # past argument=value
    ;;

    -fm=*|--fcn_mini_batch_size=*)
    fMBS="${i#*=}"
    shift # past argument=value
    ;;
    -fn=*|--fcn_num_batch=*)
    FNB="${i#*=}"
    shift # past argument=value
    ;;

    --default)
    DEFAULT=YES
    shift # past argument with no value
    ;;
    *)
            # unknown option
    ;;
esac
done

sudo rm -f output_fcn5_2gpu.log
sudo rm -f output_fcn8_2gpu.log

python multi_gpu_benchmark.py --gpu 0,1 --arch fcn5 --batch-size ${FMBS} --num-batches ${FNB} 2>&1 | tee output_fcn5_2gpu.log
python multi_gpu_benchmark.py --gpu 0,1 --arch fcn8 --batch-size ${FMBS} --num-batches ${FNB} 2>&1 | tee output_fcn8_2gpu.log

sudo rm -f output_fcn5_4gpu.log
sudo rm -f output_fcn8_4gpu.log

python multi_gpu_benchmark.py --gpu 0,1,2,3 --arch fcn5 --batch-size ${FMBS} --num-batches ${FNB} 2>&1 | tee output_fcn5_4gpu.log
python multi_gpu_benchmark.py --gpu 0,1,2,3 --arch fcn8 --batch-size ${FMBS} --num-batches ${FNB} 2>&1 | tee output_fcn8_4gpu.log
