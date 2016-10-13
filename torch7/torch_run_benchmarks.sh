#!/bin/bash

source $(pwd)'/torch/install/bin/torch-activate'
echo $PATH
which th

CMBS=16
CNB=100

RMBS=128
RNB=100

FMBS=256
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

th benchmark.lua -arch alexnet -batchSize ${CMBS} -nIterations ${CNB} -deviceId 0 2>&1 | tee output_alexnet.log
th benchmark.lua -arch resnet -batchSize ${CMBS} -nIterations ${CNB} -deviceId 0 2>&1 | tee output_resnet.log
th benchmark.lua -arch fcn5 -batchSize ${FMBS} -nIterations ${FNB} -deviceId 0 2>&1 | tee output_fcn5.log
th benchmark.lua -arch fcn8 -batchSize ${FMBS} -nIterations ${FNB} -deviceId 0 2>&1 | tee output_fcn8.log

luarocks install rnn
luarocks install dataload
pushd ./rnn/lstm/
th  lstm.lua --seqlen 32 --batchsize ${RMBS} --iters ${RNB} --hiddensize '{256,256}' --cuda --lstm --startlr 1 --cutoff 5 --maxepoch 1 --device 1 2>&1 | tee ../../output_lstm32.log
th  lstm.lua --seqlen 64 --batchsize ${RMBS} --iters ${RNB} --hiddensize '{256,256}' --cuda --lstm --startlr 1 --cutoff 5 --maxepoch 1 --device 1 2>&1 | tee ../../output_lstm64.log
popd
