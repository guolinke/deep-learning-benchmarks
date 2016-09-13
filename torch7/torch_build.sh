#!/bin/bash

sudo rm -rf torch
git clone https://github.com/torch/distro.git torch --recursive 
cd torch; bash install-deps
sudo /usr/bin/yes | ./install.sh 
source ~/.bashrc

cd ..
