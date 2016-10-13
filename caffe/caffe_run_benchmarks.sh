#!/bin/bash

BASEDIR=$(pwd)

export PYTHONPATH=$PYTHONPATH:$BASEDIR/caffe/python


CMBS=64
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


sudo rm -f output_alexnet.log
sudo rm -f output_resnet.log
sudo rm -f output_fcn5.log
sudo rm -f output_fcn8.log

cd fcn
python createFakeDataForCaffeFCN.py ${FMBS}
cd ..

cd cnn
python createFakeImageNetForCaffeCNN.py ${CMBS}
cd ..

python Generate_FCN5.py ${FMBS}
python Generate_FCN8.py ${FMBS}
python Generate_alexnet.py ${CMBS}
python Generate_resnet.py ${CMBS}

cd fcn
../caffe/build/tools/caffe train -solver=fcn5-solver.prototxt -gpu=0 2>&1 | tee ../output_fcn5.log
../caffe/build/tools/caffe train -solver=fcn8-solver.prototxt -gpu=0 2>&1 | tee ../output_fcn8.log
cd ..

cd cnn
../caffe/build/tools/caffe train -solver=alexnet-solver.prototxt -gpu=0  2>&1 | tee ../output_alexnet.log
../caffe/build/tools/caffe train -solver=resnet-solver.prototxt -gpu=0  2>&1 | tee ../output_resnet.log
cd ..
