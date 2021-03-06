#!/bin/bash

CMBS=16
CNB=40

RMBS=128
RNB=40

FMBS=8192
FNB=40

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
python createDataForCNTKFCN.py ${FMBS}
python createLabelMapForCNTKFCN.py
cd ..


sudo rm -rf Output

mpiexec -n 2 ${CNTK_HOME}/cntk configFile=fcn/fcn5.cntk parallelTrain=true configName=fcn5_2gpu minibatchSize=${FMBS} epochSize=$((${FMBS}*${FNB})) DataDir=fcn 
mpiexec -n 2 ${CNTK_HOME}/cntk configFile=fcn/fcn8.cntk parallelTrain=true configName=fcn8_2gpu minibatchSize=${FMBS} epochSize=$((${FMBS}*${FNB})) DataDir=fcn 

sudo rm -rf Output

mpiexec -n 4 ${CNTK_HOME}/cntk configFile=fcn/fcn5.cntk parallelTrain=true configName=fcn5_4gpu minibatchSize=${FMBS} epochSize=$((${FMBS}*${FNB})) DataDir=fcn 
mpiexec -n 4 ${CNTK_HOME}/cntk configFile=fcn/fcn8.cntk parallelTrain=true configName=fcn8_4gpu minibatchSize=${FMBS} epochSize=$((${FMBS}*${FNB})) DataDir=fcn 
