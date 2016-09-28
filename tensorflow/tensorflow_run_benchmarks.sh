#!/bin/bash

CMBS=16
CNB=100

RMBS=128
RNB=100

FMBS=64
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

sudo rm -f output_alexnet.log
sudo rm -f output_resnet.log
sudo rm -f output_fcn5.log
sudo rm -f output_fcn8.log
sudo rm -f output_lstm32.log
sudo rm -f output_lstm64.log

python benchmark.py --arch fcn5 --batch-size ${FMBS} --num-batches ${FNB} 2>&1 | tee output_fcn5.log
python benchmark.py --arch fcn8 --batch-size ${FMBS} --num-batches ${FNB} 2>&1 | tee output_fcn8.log
python benchmark.py --arch alexnet --batch-size ${CMBS} --num-batches ${CNB} 2>&1 | tee output_alexnet.log
python benchmark.py --arch resnet --batch-size ${CMBS} --num-batches ${CNB} 2>&1 | tee output_resnet.log
python rnn/lstm/lstm.py --batchsize ${RMBS} --iters ${RNB} --seqlen 32 --numlayer 2 --hiddensize 256 --device 0 --data_path ../cntk/rnn/PennTreebank/Data 2>&1 | tee output_lstm32.log
python rnn/lstm/lstm.py --batchsize ${RMBS} --iters ${RNB} --seqlen 64 --numlayer 2 --hiddensize 256 --device 0 --data_path ../cntk/rnn/PennTreebank/Data 2>&1 | tee output_lstm64.log
 