#!/bin/bash


mpi_home=/opt/openmpi-1.10.3

if [ ! -d "$mpi_home" ]; then
  sudo apt-get -y remove --auto-remove libopenmpi-dev
  wget https://www.open-mpi.org/software/ompi/v1.10/downloads/openmpi-1.10.3.tar.gz
  tar -xzvf ./openmpi-1.10.3.tar.gz
  cd openmpi-1.10.3
  ./configure --prefix=${mpi_home}
  make -j all
  sudo make install
  cd ..
fi

export PATH=${mpi_home}/bin:$PATH
export LD_LIBRARY_PATH=${mpi_home}/lib:$LD_LIBRARY_PATH


sudo rm -rf cntk
wget https://cntk.ai/BinaryDrop/CNTK-1-7-Linux-64bit-GPU-1bit-SGD.tar.gz
tar -zxf CNTK-1-7-Linux-64bit-GPU-1bit-SGD.tar.gz
