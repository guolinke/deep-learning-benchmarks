#!/bin/bash

MBS=64
NB=100
for i in "$@"
do
case $i in
    -m=*|--mini_batch_size=*)
    MBS="${i#*=}"
    shift # past argument=value
    ;;
    -n=*|--num_batch=*)
    NB="${i#*=}"
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
echo "Mini Batch Size  = ${MBS}"
echo "Number of Batch     = ${NB}"


sudo apt-get install -y python-pip python-dev python-numpy swig python-dev python-wheel
sudo rm -rf tensorflow
git clone https://github.com/tensorflow/tensorflow
cd tensorflow
sudo ./configure
bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
sudo /usr/bin/yes | pip uninstall tensorflow
sudo pip install /tmp/tensorflow_pkg/tensorflow*

cd ..

python benchmark_alexnet.py   --batch_size ${MBS} --nub_batches ${NB}  2>&1 | tee ~/code/convnet-benchmarks/tensorflow/output_alexnet.log
python benchmark_overfeat.py  --batch_size ${MBS} --nub_batches ${NB}  2>&1 | tee ~/code/convnet-benchmarks/tensorflow/output_overfeat.log
python benchmark_vgg.py       --batch_size ${MBS} --nub_batches ${NB}  2>&1 | tee ~/code/convnet-benchmarks/tensorflow/output_vgga.log
python benchmark_googlenet.py --batch_size ${MBS} --nub_batches ${NB}  2>&1 | tee ~/code/convnet-benchmarks/tensorflow/output_googlenet.log

