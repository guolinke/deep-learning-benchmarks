cd caffe
./caffe_nvprof_benchmarks.sh "$@"
cd ..

cd cntk
./cntk_nvprof_benchmarks.sh "$@"
cd ..

cd mxnet
./mxnet_nvprof_benchmarks.sh "$@"
cd ..

cd tensorflow
./tensorflow_nvprof_benchmarks.sh "$@"
cd ..

cd theano
./nvprof_benchmark.sh "$@"
cd ..

cd torch7
./torch_nvprof_benchmarks.sh "$@"
cd ..

