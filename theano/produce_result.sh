#!/bin/bash

# set env var
CUR_DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $CUR_DIR

for fname in log/* ; do
    echo
    echo $fname | grep -o -P '(?<=log/)(.*)(?=.log)'
    grep "number of parameters in model" $fname 
    grep 'Forward across\|Forward-Backward' $fname
done
