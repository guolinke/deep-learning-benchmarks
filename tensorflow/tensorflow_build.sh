#!/bin/bash


sudo apt-get install -y python-pip python-dev python-numpy swig python-dev python-wheel
sudo rm -rf tensorflow
git clone https://github.com/tensorflow/tensorflow 
cd tensorflow
sudo ./configure
bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
sudo rm -rf /tmp/tensorflow_pkg/
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
sudo /usr/bin/yes | pip uninstall tensorflow
sudo pip install /tmp/tensorflow_pkg/tensorflow*

cd ..
