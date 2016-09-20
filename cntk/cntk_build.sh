#!/bin/bash
sudo rm -rf cntk
# remove old openmpi
sudo apt-get -y remove --auto-remove libopenmpi-dev

mpi_home=/opt/openmpi-1.10.3
wget https://www.open-mpi.org/software/ompi/v1.10/downloads/openmpi-1.10.3.tar.gz
tar -xzvf ./openmpi-1.10.3.tar.gz
cd openmpi-1.10.3
./configure --prefix=${mpi_home}
make -j all
sudo make install
cd ..

export PATH=${mpi_home}/bin:$PATH
export LD_LIBRARY_PATH=${mpi_home}/lib:$LD_LIBRARY_PATH


wget -q -O - https://sourceforge.net/projects/boost/files/boost/1.60.0/boost_1_60_0.tar.gz/download | tar -xzf - 
cd boost_1_60_0 
./bootstrap.sh --prefix=/usr/local/boost-1.60.0
sudo ./b2 -d0 -j"$(nproc)" install
cd ..

git clone --recursive https://github.com/Microsoft/cntk/
cd cntk
git reset --hard cac191c8c3c08e546c9af25236d368c0ed2812c2
mkdir build -p
cd build
../configure --1bitsgd=yes
make -j all

cd ..
