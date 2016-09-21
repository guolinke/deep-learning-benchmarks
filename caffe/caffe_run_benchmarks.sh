
BASEDIR=$(pwd)

export PYTHONPATH=$PYTHONPATH:$BASEDIR/caffe/python

sudo rm -f output_alexnet.log
sudo rm -f output_resnet.log
sudo rm -f output_fcn5.log
sudo rm -f output_fcn8.log

cd fcn
python createFakeDataForCaffeFCN.py
cd ..

cd cnn
python createFakeImageNetForCaffeCNN.py
cd ..

cd fcn
../caffe/build/tools/caffe train -solver=ffn26752-b64-solver-GPU.prototxt -gpu=0 2>&1 | tee ../output_fcn5.log
../caffe/build/tools/caffe train -solver=ffn26752l6-b64-solver-GPU.prototxt -gpu=0 2>&1 | tee ../output_fcn8.log
cd ..

cd cnn
cd alexnet
../../caffe/build/tools/caffe train -solver=alexnet-b16-solver-GPU.prototxt -gpu=0 -iterations=100 2>&1 | tee ../../output_alexnet.log
cd ..
cd resnet
../../caffe/build/tools/caffe train -solver=resnet-b16-solver-GPU.prototxt -gpu=0 -iterations=100 2>&1 | tee ../../output_resnet.log
cd ..
cd ..
