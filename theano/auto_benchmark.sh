#!/bin/bash

# set env var
CUR_DIR="$( cd "$(dirname "$0")" ; pwd -P )"

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



# run benchmark
cd $CUR_DIR

run_benchmark () {
    mkdir -p log
    cat ~/.theanorc >> log/${1}.log
    python benchmark.py -a $1 -B $2 -n $3 >> log/${1}.log 2>&1
}

run_benchmark alexnet $CMBS $CNB
run_benchmark resnet $CMBS $CNB
run_benchmark fcn5 $FMBS $FNB
run_benchmark fcn8 $FMBS $FNB
run_benchmark lstm32 $RMBS $RNB
run_benchmark lstm64 $RMBS $RNB


# grep the result
# sh produce_result.sh > result.txt
