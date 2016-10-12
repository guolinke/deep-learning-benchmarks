#!/bin/bash

CMBS=16
CNB=100

RMBS=128
RNB=100

FMBS=64
FNB=100

CNTK_HOME=cntk/cntk/bin

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
    FMBS="${i#*=}"
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

cd fcn
python createDataForCNTKFCN.py
python createLabelMapForCNTKFCN.py
cd ..

cd cnn
python createFakeImageNetDataForCNTKCNN.py
python createLabelMapForCNTKCNN.py
cd ..

sudo rm -rf Output
sudo rm -f output_alexnet_Train.log
sudo rm -f output_resnet_Train.log
sudo rm -f output_fcn5_Train.log
sudo rm -f output_fcn8_Train.log
sudo rm -f output_lstm32_Train.log
sudo rm -f output_lstm64_Train.log

${CNTK_HOME}/cntk configFile=fcn/fcn5.cntk configName=fcn5 minibatchSize=${FMBS} epochSize=$((${FMBS}*${FNB})) DataDir=fcn 
${CNTK_HOME}/cntk configFile=fcn/fcn8.cntk configName=fcn8 minibatchSize=${FMBS} epochSize=$((${FMBS}*${FNB})) DataDir=fcn 
${CNTK_HOME}/cntk configFile=cnn/alexnet/alexnet.cntk configName=alexnet deviceId=0 minibatchSize=${CMBS} epochSize=$((${CMBS}*${CNB})) DataDir=cnn ConfigDir=cnn/alexnet
${CNTK_HOME}/cntk configFile=cnn/resnet/resnet.cntk configName=resnet deviceId=0 minibatchSize=${CMBS} epochSize=$((${CMBS}*${CNB})) DataDir=cnn ConfigDir=cnn/resnet 
${CNTK_HOME}/cntk configFile=rnn/PennTreebank/Config/rnn.cntk configName=lstm32 deviceId=0 minibatchSize=$((32*${RMBS})) epochSize=$((32*${RMBS}*${RNB})) DataDir=rnn/PennTreebank/Data ConfigDir=rnn/PennTreebank/Config trainFile=ptb.train.32.ctf 
${CNTK_HOME}/cntk configFile=rnn/PennTreebank/Config/rnn.cntk configName=lstm64 deviceId=0 minibatchSize=$((64*${RMBS})) epochSize=$((64*${RMBS}*${RNB})) DataDir=rnn/PennTreebank/Data ConfigDir=rnn/PennTreebank/Config trainFile=ptb.train.64.ctf 
