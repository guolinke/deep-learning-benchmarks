#!/bin/bash

# configure and install
sudo pip install https://github.com/theano/Theano/archive/master.zip
sudo pip install https://github.com/Lasagne/Lasagne/archive/master.zip
sudo chown -R $USER: ~/.theano/

cat <<EOF > ~/.theanorc
[global]
device = gpu
floatX = float32
optimizer_including = unsafe

[cuda]
root = /usr/local/cuda-7.5/

[lib]
cnmem = 0.45

[dnn.conv]
algo_fwd = time_once
algo_bwd_filter = time_once
algo_bwd_data = time_once
EOF


echo "Current theano version:"
python -c "import theano; print(theano.version.version)"
