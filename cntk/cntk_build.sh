#!/bin/bash
sudo rm -rf cntk
git clone --recursive https://github.com/Microsoft/cntk/
cd cntk
mkdir build -p
cd build
../configure --1bitsgd=yes
make -j all

cd ..
