#!/bin/bash
sudo rm -rf cntk
git clone https://github.com/Microsoft/cntk
cd cntk
mkdir build -p
cd build
../configure
make -j all

cd ..