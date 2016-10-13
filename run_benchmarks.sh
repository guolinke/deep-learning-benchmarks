cd caffe
./caffe_run_benchmarks.sh "$@"
cd ..

cd cntk
./cntk_run_benchmarks.sh "$@"
cd ..

cd mxnet
./mxnet_run_benchmarks.sh "$@"
cd ..

cd tensorflow
./tensorflow_run_benchmarks.sh "$@"
cd ..

cd theano
./auto_benchmark.sh "$@"
cd ..

cd torch7
./torch_run_benchmarks.sh "$@"
cd ..

