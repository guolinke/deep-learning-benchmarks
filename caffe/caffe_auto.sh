#!/bin/bash

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


sudo ./caffe_build.sh
./caffe_run_benchmarks.sh --cnn_mini_batch_size ${CMBS} --cnn_num_batch ${CNB} --rnn_mini_batch_size ${RMBS} --rnn_num_batch ${RNB} --fcn_mini_batch_size ${FMBS} --fcn_num_batch ${FNB}
