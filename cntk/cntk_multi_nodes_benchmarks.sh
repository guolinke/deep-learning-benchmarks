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


mpiexec -n 8 -hostfile hosts --map-by node ${CNTK_HOME}/cntk configFile=fcn/fcn5.cntk parallelTrain=true configName=fcn5_4gpu_2node minibatchSize=${FMBS} epochSize=$((${FMBS}*${FNB})) DataDir=fcn 
mpiexec -n 8 -hostfile hosts --map-by node ${CNTK_HOME}/cntk configFile=fcn/fcn8.cntk parallelTrain=true configName=fcn8_4gpu_2node minibatchSize=${FMBS} epochSize=$((${FMBS}*${FNB})) DataDir=fcn 
