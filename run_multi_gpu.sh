cd caffe
./caffe_multi_gpu_benchmarks.sh "$@"
cd ..

cd cntk
./cntk_multi_gpu_benchmarks.sh "$@"
cd ..

cd tensorflow
./tensorflow_multi_gpu_benchmarks.sh "$@"
cd ..

cd torch7
./torch_multi_gpu_benchmarks.sh "$@"
cd ..

