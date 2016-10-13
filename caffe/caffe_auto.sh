#!/bin/bash

sudo ./caffe_build.sh
./caffe_run_benchmarks.sh "$@"