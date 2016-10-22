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


nvprof ${CNTK_HOME}/cntk configFile=fcn/fcn5.cntk configName=prof_fcn5 minibatchSize=${FMBS} epochSize=$((${FMBS}*${FNB})) DataDir=fcn 2>&1 | tee cntk_prof_fcn5.log 
nvprof ${CNTK_HOME}/cntk configFile=fcn/fcn8.cntk configName=prof_fcn8 minibatchSize=${FMBS} epochSize=$((${FMBS}*${FNB})) DataDir=fcn  2>&1 | tee cntk_prof_fcn8.log
