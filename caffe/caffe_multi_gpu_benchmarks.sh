#!/bin/bash

BASEDIR=$(pwd)

export PYTHONPATH=$PYTHONPATH:$BASEDIR/caffe/python


CMBS=16
CNB=100

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

# 2 gpus

sudo rm -f output_alexnet_2gpu.log
sudo rm -f output_resnet_2gpu.log

cd fcn
python createFakeDataForCaffeFCN.py ${FMBS}
cd ..


python Generate_FCN5.py ${FMBS} 2
python Generate_FCN8.py ${FMBS} 2

cd fcn
../caffe/build/tools/caffe train -solver=fcn5-solver.prototxt -gpu=0,1 2>&1 | tee ../output_fcn5_2gpu.log
../caffe/build/tools/caffe train -solver=fcn8-solver.prototxt -gpu=0,1 2>&1 | tee ../output_fcn8_2gpu.log
cd ..



# 4 gpus

sudo rm -f output_fcn5_4gpu.log
sudo rm -f output_fcn8_4gpu.log


python Generate_FCN5.py ${FMBS} 4
python Generate_FCN8.py ${FMBS} 4

cd fcn
../caffe/build/tools/caffe train -solver=fcn5-solver.prototxt -gpu=0,1,2,3 2>&1 | tee ../output_fcn5_4gpu.log
../caffe/build/tools/caffe train -solver=fcn8-solver.prototxt -gpu=0,1,2,3 2>&1 | tee ../output_fcn8_4gpu.log
cd ..
