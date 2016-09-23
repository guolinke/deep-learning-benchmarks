#!/bin/bash


sudo apt-get update
sudo apt-get install -y build-essential git libatlas-base-dev libopencv-dev

sudo rm -rf mxnet 

git clone --recursive https://github.com/dmlc/mxnet
cd mxnet
cd make
echo "USE_CUDA = 1" >> config.mk
echo "USE_CUDA_PATH = /usr/local/cuda" >> config.mk
echo "USE_CUDNN = 1" >> config.mk
echo "USE_NVRTC = 1" >> config.mk
cd ..
make -j$(nproc)

sudo apt-get install -y python-setuptools
sudo apt-get install -y python-numpy

cd python; sudo python setup.py install
