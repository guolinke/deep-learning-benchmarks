cd caffe
./caffe_auto.sh "$@"
cd ..

cd cntk
./cntk_auto.sh "$@"
cd ..

cd tensorflow
./tensorflow_auto.sh "$@"
cd ..

cd theano
./theano_auto.sh "$@"
cd ..

cd torch7
./torch_auto.sh "$@"
cd ..

