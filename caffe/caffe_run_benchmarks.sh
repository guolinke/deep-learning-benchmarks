
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

caffe/build/tools/caffe train -solver=fcn/ffn26752-b64-solver-GPU.prototxt -gpu=0 2>&1 | tee output_fcn5.log
caffe/build/tools/caffe train -solver=fcn/ffn26752l6-b64-solver-GPU.prototxt -gpu=0 2>&1 | tee output_fcn8.log
caffe/build/tools/caffe train -solver=cnn/alexnet/alexnet-b16-solver-GPU.prototxt -gpu=0 -iterations=100 2>&1 | tee output_alexnet.log
caffe/build/tools/caffe train -solver=cnn/resnet/resnet-b16-solver-GPU.prototxt -gpu=0 -iterations=100 2>&1 | tee output_resnet.log
