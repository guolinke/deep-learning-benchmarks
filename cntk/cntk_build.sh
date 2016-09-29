#!/bin/bash

sudo rm -rf cntk
sudo rm -rf CNTK-1-7-1-Linux-64bit-GPU-1bit-SGD.tar.gz
wget https://cntk.ai/BinaryDrop/CNTK-1-7-1-Linux-64bit-GPU-1bit-SGD.tar.gz
sudo tar -zxf CNTK-1-7-1-Linux-64bit-GPU-1bit-SGD.tar.gz
