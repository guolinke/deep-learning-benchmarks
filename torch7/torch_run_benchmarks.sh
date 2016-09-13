#!/bin/bash

source $(pwd)'/torch/install/bin/torch-activate'
echo $PATH
which th

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

pushd ./cnn/alexnet/
th alexnetbm.lua -nEpochs 4 -batchSize ${CMBS} -nIterations ${CNB} -deviceId 1 2>&1 | tee ../../output_alexnet.log
popd

pushd ./cnn/resnet/
#th resnet.lua -depth 50 -deviceId 1 -batchSize ${CMBS} -nEpochs 4 -nIterations ${CNB} -dataset imagenet -data /home/data/ILSVRC2015/Data/CLS-LOC/ 2>&1 | tee ../../output_resnet.log
th resnet.lua -depth 50 -deviceId 1 -batchSize ${CMBS} -nEpochs 4 -nIterations 10 -dataset imagenet 2>&1 | tee ../../output_resnet.log
popd

pushd ./fcn/
th  ffn26752bm.lua   -deviceId 1 -batchSize ${FMBS} -nEpochs 2 -nIterations ${FNB} 2>&1 | tee ../output_fcn5.log
th  ffn26752l6bm.lua -deviceId 1 -batchSize ${FMBS} -nEpochs 2 -nIterations ${FNB} 2>&1 | tee ../output_fcn8.log
popd 

luarocks install rnn
luarocks install dataload
pushd ./rnn/lstm/
th  lstm.lua --seqlen 32 --batchsize ${RMBS} --iters ${RNB} --hiddensize '{256,256}' --cuda --lstm --startlr 1 --cutoff 5 --maxepoch 1 --device 1 2>&1 | tee ../../output_lstm32.log
th  lstm.lua --seqlen 64 --batchsize ${RMBS} --iters ${RNB} --hiddensize '{256,256}' --cuda --lstm --startlr 1 --cutoff 5 --maxepoch 1 --device 1 2>&1 | tee ../../output_lstm64.log
popd
