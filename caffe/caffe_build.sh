sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install -y --no-install-recommends libboost-all-dev
sudo apt-get install -y libatlas-base-dev
sudo apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev
sudo rm -rf caffe
git clone https://github.com/BVLC/caffe
cd caffe
cp Makefile.config.example Makefile.config
echo "USE_CUDNN := 1 " >> Makefile.config
make -j all
make -j pycaffe
cd ..
sudo pip install lmdb
sudo pip install scikit-image
