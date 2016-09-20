#!/bin/bash
sudo rm -rf cntk
# remove old openmpi
#sudo apt-get remove --auto-remove libopenmpi-dev
sudo apt-get install libbz2-dev

wget -q -O - https://sourceforge.net/projects/boost/files/boost/1.60.0/boost_1_60_0.tar.gz/download | tar -xzf - 
cd boost_1_60_0 
./bootstrap.sh --prefix=/usr/local/boost-1.60.0
sudo ./b2 -d0 -j"$(nproc)" install  

make -j all
git clone --recursive https://github.com/Microsoft/cntk/
cd cntk
mkdir build -p
cd build
../configure --1bitsgd=yes
make -j all

cd ..
