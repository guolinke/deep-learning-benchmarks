#!/bin/bash

# set env var
CUR_DIR="$( cd "$(dirname "$0")" ; pwd -P )"

CMBS=16
CNB=70

RMBS=128
RNB=70

FMBS=8192
FNB=70

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
sudo rm -rf log
run_benchmark () {
    mkdir -p log
    cat ~/.theanorc > log/theano_prof_${1}.log
    nvprof python benchmark.py -a $1 -B $2 -n $3 2>&1 | tee log/theano_prof_${1}.log 
}

run_benchmark fcn5 $FMBS $FNB
run_benchmark fcn8 $FMBS $FNB


# grep the result
# sh produce_result.sh > result.txt
