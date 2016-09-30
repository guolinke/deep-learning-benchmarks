#!/bin/bash

sudo rm -rf torch
git clone https://github.com/torch/distro.git torch --recursive 
git reset --hard a86fdb060509234288c8b2c5eae2d3d66b51901a
cd torch; bash install-deps
sudo /usr/bin/yes | ./install.sh 
source ~/.bashrc

cd ..
